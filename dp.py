import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import copy
from pprint import pprint
from model import Layer, ModelArgs, Transformer
from torch.profiler import profile, record_function, ProfilerActivity

from utils import (
    loss_fn,
    configure_optimizers,
    grad_to_tensor,
    tensor_to_grad,
    init_tensor,
    print_rank,
    params,
)


class DP:
    def __init__(self, config, batch_size, gradient_accumulation_steps=1):
        # Setup world info

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        config.n_layers *= self.world_size
        self.config = config

        self.model_fp32 = Layer(self.rank, self.world_size, self.config).float().cuda()

        self.models = [copy.deepcopy(self.model_fp32).bfloat16()]

        self.n_parameter = sum(x.numel() for x in params(self.models[0]))

        self.buffers = {
            "weight": init_tensor(self.n_parameter, init_func=torch.zeros),
            "grad": init_tensor(self.n_parameter, init_func=torch.zeros),
        }

        self.loss_fn = loss_fn

        self.optimizer = configure_optimizers(self.model_fp32)
        self.optimizer.zero_grad()

        self.flattern_weight()
        self.counter = {}
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def flattern_weight(self):
        i = 0
        for p in params(self.models[0]):
            n = p.data.numel()
            self.buffers["weight"][i : i + n] = p.data.view(-1)
            i += n

        i = 0
        for p in params(self.models[0]):
            n = p.data.numel()
            p.data = self.buffers["weight"][i : i + n].view(p.data.shape)
            i += n

    # mark
    def forward_backward_step(self, inputs, targets=None):
        gradient_accumulation_steps = self.gradient_accumulation_steps
        bsz, seq_len = inputs.shape
        micro_bsz = bsz // gradient_accumulation_steps

        inputs = torch.split(inputs, micro_bsz, 0)
        targets = torch.split(targets, micro_bsz, 0)

        for i in range(gradient_accumulation_steps):
            x, y = inputs[i], targets[i]
            y_ = self.models[0].forward(x)
            loss = self.loss_fn(y_, y)
            loss.backward()

        grad_to_tensor(self.models[0], self.buffers["grad"])
        dist.all_reduce(self.buffers["grad"])

        self.update()
        return loss

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def update(self):
        tensor_to_grad(
            self.buffers["grad"] / self.world_size / self.gradient_accumulation_steps,
            self.model_fp32,
        )
        self.buffers["grad"].zero_()

        nn.utils.clip_grad_norm_(params(self.model_fp32), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        i = 0
        for p in params(self.model_fp32):
            n = p.data.numel()
            self.buffers["weight"][i : i + n] = p.data.view(-1).bfloat16()
            i += n
