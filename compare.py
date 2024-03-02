import torch
import os
import math
from functools import partial

import copy
from utils import update, loss_fn, configure_optimizers, get_lr, save_model
from torch import nn
from tinystories import Task
import time


from model import Transformer, ModelArgs, Layer

torch.manual_seed(1234)

batch_size = 64
n_layers = 6

model_args = dict(
    dim=288,
    n_heads=6,
    n_kv_heads=None,
    vocab_size=32000,
    multiple_of=32,
    max_seq_len=128,
    dropout=0.0,
    n_layers=n_layers,
)
config = ModelArgs(**model_args)

transformer = Transformer(config)

iter_batches = partial(
    Task.iter_batches,
    batch_size=batch_size,
    max_seq_len=config.max_seq_len,
    device="cuda",
    num_workers=0,
)


eval_interval = 100
eval_iters = 100
gradient_accumulation_steps = 4
log_interval = 10


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        batch_iter = iter_batches(split)
        losses = torch.zeros(eval_iters)  # keep on CPU
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            Y_ = model(X)
            loss = loss_fn(Y_, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Model(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for l in layers:
            self.layers.append(l)

    def forward(self, x):
        x = self.layers[0](x, is_first=True)
        for l in self.layers[1:-1]:
            x = l(x)
        x = self.layers[-1](x, is_last=True)
        return x


if __name__ == "__main__":
    learning_rate = 5e-4
    torch.manual_seed(1234)

    config.n_layers = 1
    layers = [Layer(i, 1, config) for i in range(n_layers)]
    config.n_layers = 6
    # layers[0].tok_embeddings.weight = layers[-1].output.weight

    model_fp32 = Model(layers).cuda()
    model = copy.deepcopy(model_fp32).bfloat16()

    optimizer = configure_optimizers(model_fp32)
    optimizer.zero_grad()

    train_batch_iter = iter_batches("train")
    X, Y = next(train_batch_iter)

    iter_num = 0
    tokens_per_iter = gradient_accumulation_steps * batch_size * config.max_seq_len
    start = time.time()
    while True:
        lr = get_lr(learning_rate, iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if (iter_num + 1) % eval_interval == 0:
            losses = estimate_loss(model)
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            save_model(
                layers=layers,
                optimizer=optimizer,
                model_args=model_args,
                iter_num=iter_num,
            )

        for _ in range(gradient_accumulation_steps):
            Y_ = model(X)
            loss = loss_fn(Y_, Y)
            loss.backward()

            X, Y = next(train_batch_iter)

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        update(model_fp32, model, optimizer)
        model = copy.deepcopy(model_fp32).bfloat16()

        if iter_num % 1 == 0:
            # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
            lossf = loss.item()
            print(
                f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | time {time.time()-start }"
            )
        iter_num += 1
