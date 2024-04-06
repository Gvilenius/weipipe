"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU small debug run, example:
$ python -m train.py --compile=False --eval_iters=10 --batch_size=8

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import torch.nn.functional as F
import math
import deepspeed as ds
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import torch
from model import Transformer, ModelArgs
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from tinystories import Task
import json

with open("config.json", "r") as f:
    config = json.load(f)

# -----------------------------------------------------------------------------
# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 100
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume'
# data
batch_size = config[
    "batch_size"
]  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = config["max_seq_len"]
dataset = "tinystories"  # tinystories|tinyshakespeare
# model
dim = config["dim"]
n_layers = config["n_layers"]
n_heads = config["n_heads"]
multiple_of = config["multiple_of"]
dropout = config["dropout"]
# adamw optimizer
gradient_accumulation_steps = config[
    "gradient_accumulation_steps"
]  # used to simulate larger batch sizes
learning_rate = config["lr"]  # max learning rate
max_iters = config["iter_nums"]  # total number of training iterations

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
# exec(open("configurator.py").read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

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

gradient_accumulation_steps //= ddp_world_size
batch_size //= gradient_accumulation_steps

tokens_per_iter = (
    gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
)

if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    print(
        f"breaks down as: {gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {batch_size} batch size * {max_seq_len} max seq len"
    )

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]


# task-specific setup
task = Task
iter_batches = partial(
    task.iter_batches,
    batch_size=batch_size,
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
    n_kv_heads=n_heads,
    vocab_size=config["vocab_size"],
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
)  # start with model_args from command line
gptconf = ModelArgs(**model_args)
model = Transformer(gptconf)

model.to(device)
# initialize a GradScaler. If enabled=False scaler is a no-op

# Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
# construction time since NCCL does not support `ComplexFloat`
prefix = "_orig_mod." if compile else ""
model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}

ds_config = {
    "train_micro_batch_size_per_gpu": batch_size,
    # "amp": {"enabled": True, "opt_level": "O2"},
    "optimizer": {
        "type": "Adam",
        "params": {"lr": 5e-4, "betas": [beta1, beta2], "weight_decay": weight_decay},
    },
    "gradient_clipping": grad_clip,
    "bf16": {
        "enabled": True,
    },
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "zero_optimization": {
        "stage": 3,
    },
}
# optimizer
# optimizer = model.configure_optimizers(
#     weight_decay, learning_rate, (beta1, beta2), device_type
# )

model, _, _, _ = ds.initialize(
    model=model, model_parameters=model.parameters(), config=ds_config
)
model.train()


checkpoint = None  # free up memory


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
while iter_num < config["iter_nums"]:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for i in range(gradient_accumulation_steps):
        loss = model(X, Y)
        model.backward(loss)
        model.step()
    # immediately async prefetch next batch while model is doing the forward pass on the GPU
    X, Y = next(train_batch_iter)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
        lossf = loss.item()
        print(f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

    if iter_num == 0:
        print(
            f"rank{ddp_local_rank} memory used: {torch.cuda.max_memory_allocated()/1024**3}G"
        )

if config["output"] and torch.distributed.get_rank() == 0:
    with open("result-ds", "a") as f:
        f.write(
            f'{config["batch_size"]}-{config["gradient_accumulation_steps"]}: {dt:.2f}\n'
        )
