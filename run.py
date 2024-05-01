from weipipe import WeiPipe
from actpipe import ActPipe
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
from utils import get_lr, print_rank
from model import ModelArgs
from torch.profiler import profile, record_function, ProfilerActivity

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

with open("config.json", "r") as f:
    config = json.load(f)

model_args = dict(
    dim=config["dim"],
    n_heads=config["n_heads"],
    n_kv_heads=None,
    vocab_size=config["vocab_size"],
    multiple_of=config["multiple_of"],
    max_seq_len=config["max_seq_len"],
    dropout=config["dropout"],
    n_layers=config["n_layers"],
)

model_size = (
    12 * config["n_layers"] * config["dim"] ** 2 * 2 / dist.get_world_size() / 1024**3
)
optimizer_size = (2 + 2 + 2 + 1) * model_size

data_size = config["batch_size"] * config["max_seq_len"] * 2 * 2 / 1024**3

# share
vocab_size = config["vocab_size"] * config["dim"] * 2 / 1024**3
vocab_optimier_size = (2 + 2 + 2 + 1) * vocab_size

print_rank(0, f"model     {model_size:.4f} G")
print_rank(0, f"optimizer {optimizer_size:.4f} G")
print_rank(0, f"data_size {data_size: .4f}  G")
print_rank(0, f"vocab     {vocab_size: .4f} G")
print_rank(0, f"vocab opt {vocab_optimier_size: .4f} G")


# microbatch size
batch_size = config["batch_size"]
learning_rate = config["lr"]
mode = config["mode"]

gradient_accumulation_steps = config["gradient_accumulation_steps"]
assert gradient_accumulation_steps % dist.get_world_size() == 0

strategy = {
    "act": ActPipe,
    "wei": WeiPipe,
}

model = strategy[mode](
    ModelArgs(**model_args),
    batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps // dist.get_world_size(),
)

eval_interval = 500
iter_batches = partial(
    Task.iter_batches,
    batch_size=batch_size,
    max_seq_len=model_args["max_seq_len"],
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
    start = time.time()
    n_total_samples = 0

    # while n_total_samples < 64 * 100 + 1:
    prof = profile(
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=2, repeat=1),
        record_shapes=False,
        with_stack=False,
        # activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    )
    prof.start()
    while iter_num < config["iter_nums"]:
        prof.step()
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

        loss = model.forward_backward_step(X, Y)
        # print(prof.key_averages().table(sort_by="cuda_time"))

        X, Y = next(train_batch_iter)

        if mode != "act":
            loss_rank = 0
            n_total_samples += batch_size * dist.get_world_size()
        else:
            loss_rank = dist.get_world_size() - 1
            n_total_samples += batch_size

        dt = time.time() - start

        start = time.time()
        if iter_num % 1 == 0 and dist.get_rank() == loss_rank:
            # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
            print(
                f"{iter_num} | loss {loss.item():.4f} | lr {lr:e} | time {dt*1000 :.2f}ms",
            )
            # print(
            #     f"{iter_num} | lr {lr:e} | time {dt*1000 :.2f}ms",
            # )

        if iter_num < 3:
            print_rank(
                0, f"max memory used: {torch.cuda.max_memory_allocated()/1024**3:.2f}G"
            )

        iter_num += 1
    prof.stop()
    if dist.get_rank() == 1:
        prof.export_chrome_trace("trace.json")

    if config["output"] and dist.get_rank() == 0:
        with open("result-wei", "a") as f:
            f.write(
                f'{config["batch_size"]}-{config["gradient_accumulation_steps"]}: {dt:.2f}\n'
            )
