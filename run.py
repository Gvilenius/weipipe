from weipipe import WeiPipe
import time
import os
from functools import partial
import torch
from torch import nn
import torch.distributed.rpc as rpc
import tempfile
import torch.distributed as dist
import argparse
from tinystories import Task
from utils import get_lr, loss_fn

torch.manual_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--rank", default=-1, type=int)
args = parser.parse_args()


tmpfile = tempfile.NamedTemporaryFile()
rpc.init_rpc(
    name="worker",
    rank=0,
    world_size=1,
    rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
        init_method="file://{}".format(tmpfile.name),
        _transports=["ibv", "uv"],
        _channels=["cuda_ipc", "cuda_basic"],
    ),
)


def init_process_group():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())


def print_rank0(x):
    if dist.get_rank() == 0:
        print(x)


init_process_group()

model = WeiPipe()
max_seq_len = 128

learning_rate = 5e-4
gradient_accumulation_steps = 1
batch_size = 32

eval_interval = 100

iter_batches = partial(
    Task.iter_batches,
    batch_size=batch_size,
    max_seq_len=max_seq_len,
    device="cuda",
    num_workers=0,
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
    learning_rate = 5e-4
    torch.manual_seed(1234)

    train_batch_iter = iter_batches("train")
    X, Y = next(train_batch_iter)

    iter_num = 0
    tokens_per_iter = gradient_accumulation_steps * batch_size * max_seq_len
    start = time.time()
    while True:
        lr = get_lr(learning_rate, iter_num)
        model.set_lr(lr)

        if (iter_num + 1) % eval_interval == 0:
            transformer = model.get_full_transformer()

            if dist.get_rank() == 0:
                loss = estimate_loss(transformer)
                print(f" loss eval is {loss}")
                out_dir = "out"
                print(f"saving checkpoint to {out_dir}")
                transformer.export(os.path.join(out_dir, "model.bin"))

        for i in range(gradient_accumulation_steps):
            loss = model.forward_backward_step(X, Y, first=i == 0)
            X, Y = next(train_batch_iter)

        model.update()

        if iter_num % 1 == 0:
            # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
            lossf = loss.item()
            print_rank0(
                f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | time {time.time() - start :.2f}"
            )
        iter_num += 1
