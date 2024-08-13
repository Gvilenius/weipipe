import torch.distributed as dist
import torch.nn.functional as F
import math
import os
import time
from functools import partial

import torch
from model import Transformer, ModelArgs
from torch.nn.parallel import DistributedDataParallel as DDP

from tinystories import Task
import argparse
from utils import get_env
from utils import output_statistics
import numpy as np


def init_process_group():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    

parser = argparse.ArgumentParser()
parser.add_argument("--stage", default=3, type=int)
args = parser.parse_args()

# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 100
eval_only = False  # if True, script exits right after the first eval
# model
dim = get_env("HIDDEN_SIZE")
n_heads = get_env("ATTENTION_HEADS")
max_seq_len = get_env("SEQ_LEN")
n_layers = get_env("LAYERS")

gradient_accumulation_steps = get_env("ACC_STEP")
max_iters = get_env("EXIT_INTERVAL") 
micro_batch_size = get_env("MICRO_BATCH_SIZE")


multiple_of = 32
dropout = 0.0
learning_rate = 1e-5

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 0  # how many steps to warm up for
# system
device = (
    "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
)
dtype = "bfloat16"  # float32|bfloat16|float16
compile = False  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]


config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
# exec(open("configurator.py").read())  # overrides from command line or config file
# -----------------------------------------------------------------------------
init_process_group()


# fixing some hyperparams to sensible defaults
lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla
min_lr = 0.0  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?


ddp_rank = int(os.environ["RANK"])
ddp_local_rank = int(os.environ["LOCAL_RANK"])
ddp_world_size = int(os.environ["WORLD_SIZE"])
device = f"cuda:{ddp_local_rank}"
torch.cuda.set_device(device)
master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
seed_offset = ddp_rank  # each process gets a different seed
# world_size number of processes will be training simultaneously, so we can scale
# down the desired gradient accumulation iterations per process proportionally


if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler

# task-specific setup
task = Task
iter_batches = partial(
    task.iter_batches,
    batch_size=micro_batch_size,
    max_seq_len=max_seq_len,
    device=device,
    num_workers=0,
)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9


# model init
model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=None,
    vocab_size=32000,
    multiple_of=32,
    max_seq_len=max_seq_len,
    dropout=dropout,
)  # start with model_args from command line
if ddp_rank == 0:
    print(model_args)

gptconf = ModelArgs(**model_args)
model = Transformer(gptconf)


if not bool(get_env("TRAIN_EMBEDDING")):
    model.tok_embeddings.weight.requires_grad = False

model.to(device)
# initialize a GradScaler. If enabled=False scaler is a no-op

# Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
# construction time since NCCL does not support `ComplexFloat`
prefix = "_orig_mod." if compile else ""
model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}



# optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)

if ddp_rank == 0:
    print("num parameters: ", sum(p.numel() for p in model.parameters()) / 1e9, "e9")

prefix = "_orig_mod." if compile else ""
model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
model = DDP(model)

ctx = torch.amp.autocast(device_type=device_type, dtype=torch.float16)


model.train()

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# training loop
train_batch_iter = iter_batches("train")
X, Y = next(train_batch_iter)  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0

dts = []
while iter_num < max_iters:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    
    for i in range(gradient_accumulation_steps):
        with ctx:
            loss = model(X, Y)
        X, Y = next(train_batch_iter)
        loss.backward()
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
    optimizer.step()
    optimizer.zero_grad()

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    dts.append(dt * 1000)
    if iter_num % log_interval == 0 and master_process:
        # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
        lossf = loss.item()
        print(f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1

    if ddp_local_rank == 0:
        memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"rank{ddp_local_rank} max memory used: {memory:.2f}G")


# if ddp_rank == 0:
# prof.export_chrome_trace("trace.json")



t = f"{np.mean(dts[1:]):.2f}"
if torch.distributed.get_rank() == 0:
    output_statistics ("ddp", t, memory)

dist.destroy_process_group()
