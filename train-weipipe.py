from weipipe import WeiPipe
import time
import json
from functools import partial
import torch
from torch import nn
import torch.distributed.rpc as rpc
import tempfile
import torch.distributed as dist
import argparse
from tinystories import Task
from utils import get_lr, print_rank, output_statistics, get_profiler
from model import ModelArgs
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import os

torch.manual_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--rank", default=-1, type=int)
args = parser.parse_args()


# tmpfile = tempfile.NamedTemporaryFile()
# rpc.init_rpc(
#     name="worker",
#     rank=0,
#     world_size=1,
#     rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
#         init_method="file://{}".format(tmpfile.name),
#         _transports=["ibv", "uv"],
#         _channels=["cuda_ipc", "cuda_basic"],
#     ),
# )


def init_process_group():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())


init_process_group()

world_size = dist.get_world_size()

def get_env(k):
    return int(os.environ[k])

dim = get_env("HIDDEN_SIZE")
n_heads = get_env("ATTENTION_HEADS")
max_seq_len = get_env("SEQ_LEN")

n_layers = get_env("LAYERS") 

checkpointing = bool(get_env("CHECKPOINTING"))
train_embedding = bool(get_env("TRAIN_EMBEDDING"))
micro_batch_size = get_env("MICRO_BATCH_SIZE")
gradient_accumulation_steps = get_env("ACC_STEP")
iters_num = get_env("EXIT_INTERVAL")

model_args = dict(
    dim=dim,
    n_heads=n_heads,
    n_kv_heads=None,
    vocab_size=4,
    multiple_of=32,
    max_seq_len=max_seq_len,
    dropout=0.0,
    n_layers = n_layers,
    checkpointing=checkpointing,
    train_embedding = train_embedding
)

print_rank(0, model_args)

# microbatch size
iter_batches = partial(
    Task.iter_batches,
    batch_size=micro_batch_size,
    max_seq_len=max_seq_len,
    device="cuda",
    num_workers=2,
)
train_batch_iter = iter_batches("train")

model = WeiPipe(
    ModelArgs(**model_args),
    gradient_accumulation_steps=gradient_accumulation_steps,
    train_embedding=train_embedding,
    dl_iter=train_batch_iter,
    batch_size = micro_batch_size
)

@torch.no_grad()
def estimate_loss(model, eval_iters=20):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        batch_iter = iter_batches(split)
        losses = torch.zeros(eval_iters)  # keep on CPU
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            model(X, Y)
            loss = model.last_loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == "__main__":
    learning_rate = 1e-3
    model.set_lr(learning_rate)
    torch.manual_seed(1234)

    iter_num = 0
    start = time.time()
    n_total_samples = 0
    
    enable_prof = bool(int(os.environ["PROF"]))
    if enable_prof:
        prof = get_profiler()
        prof.start()
    
    dts = []
    while iter_num < iters_num:

        if enable_prof:
                prof.step()
            
        loss = model.forward_backward_step()
        loss = loss.item()
  
            
        dt = time.time() - start
        dts.append(dt * 1000)
        start = time.time() 

        # if iter_num % 1 == 0 and dist.get_rank() == loss_rank:
            # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
            # print(
            #     f"{iter_num} | loss {loss:.4f} | lr {lr:e} | time {dt*1000 :.2f}ms",
            # )
        memory = torch.cuda.max_memory_allocated() / 1024**3
        
        if dist.get_rank() == 0:
            print(
                f"{iter_num} | rank: {dist.get_rank()}   loss {loss:.4f} | time {dt*1000 :.2f}ms | memory {memory:.2f} G",
            )
        iter_num += 1
        

            
    if enable_prof:
        prof.stop()
        prof.export_chrome_trace(os.environ["WEIPIPE_DIR"]  + f"/result/weipipe-trace-{dist.get_rank()}.json")

    if dist.get_rank() == 0:
        t = np.mean(dts[2:])
        print(t)
        output_statistics("weipipe", t, memory)

    model.destroy()
