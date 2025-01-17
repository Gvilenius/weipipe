import inspect
import math
import torch
import torch.nn.functional as F
import torch.distributed as dist

import os
import json
import csv
from model import Transformer, ModelArgs
import numpy as np

def set_env(k, v):
    os.environ[k] = str(v)

def get_env(k):
    return int(os.environ[k])


def serialize_model(model, name="unnamed"):
    array = np.array([])
    for n, p in model.named_parameters():
        print(n)
        array = np.append(array, p.data.cpu().numpy())
    np.savetxt(name, array, fmt="%.6f", delimiter="\n")
    
def serialize_grad(model, name="unnamed"):
    array = np.array([])
    for n, p in model.named_parameters():
        print(n)
        array = np.append(array, p.grad.data.cpu().numpy())
    np.savetxt(name, array, fmt="%.6f", delimiter="\n")
    
def output_statistics(fname, t, memory):
    return
    world_size = dist.get_world_size()
    fname = os.environ["WEIPIPE_DIR"]  + "/result/{}.csv".format(fname)
    init = not os.path.exists (fname)
    with open(fname, "a") as f:
        writer = csv.writer(f)
        
        l = get_env ("LAYERS")
        h = get_env ("HIDDEN_SIZE")
        s = get_env ("SEQ_LEN")
        acc_step = get_env ("ACC_STEP")
        m = get_env ("MICRO_BATCH_SIZE")
        v = 32000
        memory = f"{memory:.2f}"

        nparam = (12 * l * h**2 + v*h) / 1024**2
        if init:
            writer.writerow (["nparam/M", "ngpu", "nlayer", "hidden", "seq_len", "n_micro", "mb", "time", "memory"])
        writer.writerow([nparam, world_size, l, h, s, acc_step, m, int(t), memory])

def get_lr(learning_rate, it, warmup_iters=0, lr_decay_iters=100000, min_lr=0.0):
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


def update(model_fp32, model, optimizer):
    n = sum(x.numel() for x in model.parameters())
    grad = init_tensor(n)
    grad_to_tensor(model, grad)
    tensor_to_grad(grad, model_fp32)
    optimizer.step()
    optimizer.zero_grad()


def save_model(layers, optimizer, model_args, iter_num):
    print(model_args)
    transformer = Transformer(ModelArgs(**model_args))
    for i in range(len(layers)):
        transformer.layers[i] = layers[i].layers[0]
    transformer.tok_embeddings = layers[0].tok_embeddings
    transformer.norm = layers[-1].norm
    transformer.output = layers[-1].output

    checkpoint = {
        "model": transformer.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": iter_num,
    }

    out_dir = "out"
    print(f"saving checkpoint to {out_dir}")
    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    transformer.export(os.path.join(out_dir, "model.bin"))


def loss_fn(y_, y):
    return F.cross_entropy(y_.view(-1, y_.shape[-1]), y.view(-1))

def get_profiler():
    return torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
            record_shapes=False,
            with_stack=True,
            # activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        )

def print_rank(rank, *x):
    if dist.get_rank() == rank:
        print(*x)


def params(m):
    return m.parameters()


def grad_to_tensor(model, tensor):
    i = 0
    for p in params(model):
        n = p.data.numel()
        if p.grad is not None:
            data = p.grad.flatten()
            tensor[i : i + n] += data
        i += n
    model.zero_grad()



def tensor_to_grad(tensor, model):
    i = 0
    for p in params(model):
        n = p.data.numel()
        p.grad = tensor[i : i + n].reshape(p.data.shape).float()
        i += n


def init_tensor(n, dtype=torch.float16, init_func=torch.empty):
    return init_func(n).cuda().to(dtype)


def configure_optimizers(
    model,
    weight_decay=1e-1,
    learning_rate=1e-3,
    betas=(0.9, 0.95),
    device_type="cuda",
):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    # num_decay_params = sum(p.numel() for p in decay_params)
    # num_nodecay_params = sum(p.numel() for p in nodecay_params)
    # print(
    #     f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    # )
    # print(
    #     f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    # )
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, **extra_args
    )
    print(f"using fused AdamW: {use_fused}")

    return optimizer
