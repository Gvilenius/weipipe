from weipipe import WeiPipe, ActPipe
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
from utils import get_lr, print_rank
from model import ModelArgs
from torch.profiler import profile, record_function, ProfilerActivity

torch.manual_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--rank", default=-1, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--mode", default="wei", type=str)
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


init_process_group()

model_args = dict(
    dim=144,
    n_heads=6,
    n_kv_heads=None,
    vocab_size=32000,
    multiple_of=32,
    max_seq_len=128,
    dropout=0.0,
    n_layers=6,
)

learning_rate = 5e-4
batch_size = args.batch_size

if args.mode == "act":
    model = ActPipe(ModelArgs(**model_args), batch_size=batch_size)
else:
    model = WeiPipe(ModelArgs(**model_args), batch_size=batch_size)


eval_interval = 100

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

    while n_total_samples < 64 * 100 + 1:
        lr = get_lr(learning_rate, iter_num)
        model.set_lr(lr)
        # if (iter_num + 1) % eval_interval == 0:
        #     transformer = model.get_full_transformer()

        #     if dist.get_rank() == 0:
        #         loss = estimate_loss(transformer)
        #         print(f" loss eval is {loss}")
        #         out_dir = "out"
        #         print(f"saving checkpoint to {out_dir}")
        #         transformer.export(os.path.join(out_dir, "model.bin"))
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        loss = model.forward_backward_step(X, Y)
        # print(prof.key_averages().table(sort_by="cuda_time"))
        # prof.export_chrome_trace("trace.json")

        X, Y = next(train_batch_iter)

        if args.mode == "wei":
            loss_rank = 0
            n_total_samples += batch_size * dist.get_world_size()
        else:
            loss_rank = dist.get_world_size() - 1
            n_total_samples += batch_size

        if iter_num % 5 == 0 and dist.get_rank() == loss_rank:
            # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
            print(
                f"{iter_num} | loss {loss.item():.4f} | lr {lr:e} | time {time.time() - start :.2f}",
            )
        iter_num += 1
