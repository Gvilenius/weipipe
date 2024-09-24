import torch.distributed as dist
import torch.nn.functional as F
import math
import deepspeed as ds
import os
import time
from functools import partial

import torch
from model import Transformer, ModelArgs

from tinystories import Task
import argparse
import logging

import utils

from utils import get_env, get_lr, print_rank

import numpy as np 

deepspeed_logger = logging.getLogger("DeepSpeed")
deepspeed_logger.setLevel(logging.ERROR)
for hdl in deepspeed_logger.handlers:
    hdl.setLevel(logging.ERROR)
parser = argparse.ArgumentParser()
parser.add_argument("--stage", default=3, type=int)
args = parser.parse_args()

rank = get_env("RANK")  
local_rank = get_env("LOCAL_RANK")
world_size = get_env("WORLD_SIZE") 

# -----------------------------------------------------------------------------
# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 100
eval_only = False  # if True, script exits right after the first eval

micro_batch_size = get_env("MICRO_BATCH_SIZE")

# model
dim = get_env("HIDDEN_SIZE")
n_heads = get_env("ATTENTION_HEADS")
max_seq_len = get_env("SEQ_LEN")
n_layers = get_env("LAYERS")

gradient_accumulation_steps = get_env("ACC_STEP")
max_iters = get_env("EXIT_INTERVAL") 

multiple_of = 32
dropout = 0.0

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 0  # how many steps to warm up for
# system
device = (
    "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
)

# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
# exec(open("configurator.py").read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

# fixing some hyperparams to sensible defaults
lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla
min_lr = 0.0  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# various inits, derived attributes, I/O setup


device = f"cuda:{local_rank}"
torch.cuda.set_device(device)
master_process = rank == 0  # this process will do logging, checkpointing etc.

# world_size number of processes will be training simultaneously, so we can scale
# down the desired gradient accumulation iterations per process proportionally

tokens_per_iter = (
    gradient_accumulation_steps * world_size * micro_batch_size * max_seq_len
)

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1234)
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

iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=None,
    vocab_size=32000,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
)  # start with model_args from command line

if master_process:
    print(model_args)
    
gptconf = ModelArgs(**model_args)
model = Transformer(gptconf)
# n=0
# for p in model.layers[0].parameters():
#     n+= p.numel()
# print(n,"--", 12*dim**2)
if not bool(get_env("TRAIN_EMBEDDING")):
    print("freeze_embedding")
    model.tok_embeddings.weight.requires_grad = False

model.to(device)
# initialize a GradScaler. If enabled=False scaler is a no-op


if bool(get_env("CHECKPOINTING")):
    ds.checkpointing.configure(None)

ds_config = {
    "train_micro_batch_size_per_gpu": micro_batch_size,
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 1e-3, "betas": [beta1, beta2], "weight_decay": weight_decay},
    },
    "gradient_clipping": grad_clip,
    # "bf16": {
    #     "enabled": True,
    # },
    "fp16": {
        "enabled": True,
        "loss_scale" : 1,
    },
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "zero_optimization": {
        "stage": 3,
        "contiguous_gradients": False,
        "overlap_comm": True,
        "stage3_max_live_parameters": 12 * dim**2 ,
        "stage3_max_reuse_distance": 0,
        "stage3_prefetch_bucket_size": 12 * dim**2,
        "stage3_param_persistence_threshold": 10,
    },
}

# from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_cold
# estimate_zero3_model_states_mem_needs_all_cold(model, num_gpus_per_node=2, num_nodes=1)

if rank == 0:
    print("num parameters: ", sum(p.numel() for p in model.parameters()) / 1e9, "e9")




model, _, _, _ = ds.initialize(
    model=model, model_parameters=model.parameters(), config=ds_config
)

model.train()

# training loop
train_batch_iter = iter_batches("train")
X, Y = next(train_batch_iter)  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process

running_mfu = -1.0

dts = []


enable_prof = bool(int(os.environ["PROF"]))
if enable_prof:
    prof = utils.get_profiler()
    prof.start()


while iter_num < max_iters:
    if enable_prof:
        prof.step()
    # determine and set the learning rate for this iteration
    for i in range(gradient_accumulation_steps):
        loss = model(X, Y)
        model.backward(loss)
        model.step()
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = next(train_batch_iter)

    # timing and logging
    lossf = loss.item()
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    dts.append(dt * 1000)
    if iter_num % log_interval == 0 and master_process:
        # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
        
        print(f"{iter_num} | loss {lossf:.4f} | {dt*1000:.2f}ms")
        # print(f"{iter_num} | lr {lr:e} | {dt*1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1

    if local_rank == 0:
        memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"rank{local_rank} max memory used: {memory:.2f}G")


if enable_prof:
    prof.stop()
    if dist.get_rank() == 0:
        prof.export_chrome_trace(os.environ["WEIPIPE_DIR"]  + "/ds-trace.json")



t = np.mean(dts[1:])
if dist.get_rank() == 0:
    utils.output_statistics ("ds", t, memory)

