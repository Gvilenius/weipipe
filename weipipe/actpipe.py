import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import copy
from pprint import pprint
from model import Layer, ModelArgs, Transformer

from utils import (
    loss_fn,
    configure_optimizers,
    grad_to_tensor,
    tensor_to_grad,
    model_to_tensor,
    tensor_to_model,
    init_tensor,
    print_rank0,
)


def debug(func):
    def wrapper(self, *args):
        print(f"rank {dist.get_rank()} in {func.__name__}")
        func(self, *args)
        print(f"rank {dist.get_rank()} out {func.__name__}")

    return wrapper


class ActPipe:
    def __init__(self, config) -> None:
        # Setup world info

        self.config = config
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.model_fp32 = Layer(self.rank, self.world_size, self.config).float().cuda()

        self.model = copy.deepcopy(self.model_fp32).bfloat16()

        self.loss_fn = loss_fn
        self.activations = []

        self.optimizer = configure_optimizers(self.model_fp32)

        self.batch_size = 64
        self.act_shape = (self.batch_size, self.config.max_seq_len, self.config.dim)

        self.optimizer.zero_grad()

    def forward_step(self):
        is_first = self.rank == 0
        is_last = self.rank == self.world_size - 1

        x = self.activations[-1]
        with torch.no_grad():
            y = self.model(x, is_first=is_first, is_last=is_last)
        return y

    def backward_step(self, grad):
        is_first = self.rank == self.world_size - 1
        is_last = self.rank == 0

        inputs = self.activations.pop().detach()

        if not is_first:
            inputs.requires_grad = True

        # recomputation
        outputs = self.model(inputs, is_first=is_first, is_last=is_last)
        outputs.backward(grad)

        return inputs.grad

    def calc_grad(self, targets):
        outputs = self.activations.pop()
        # outputs.requires_grad = True
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        grad = outputs.grad
        return grad, loss

    def forward_backward_step(self, inputs, targets, gradient_accumulation_steps=1):
        bsz, seq_len = inputs.shape
        micro_bsz = bsz // gradient_accumulation_steps

        inputs = torch.split(inputs, micro_bsz, 0)
        targets = torch.split(targets, micro_bsz, 0)

        for istep in range(gradient_accumulation_steps):
            input = inputs[istep]
            target = targets[istep]
            self.activations.append(input)
            if self.rank != 0:
                self.recv_act()
            y = self.forward_step()
            print(self.rank)
            if self.rank != self.world_size - 1:
                self.send_act(y)

        for istep in range(gradient_accumulation_steps):
            # if self.rank != self.world_size - 1:
            #     grad = self.recv_grad()

            grad = torch.empty(self.act_shape).bfloat16().cuda()
            if self.rank == self.world_size - 1:
                target = targets[istep]
                grad, loss = self.calc_grad(target)

            grad = self.backward_step(grad)

        if self.rank != 0:
            self.send_grad(grad)

        self.update()
        return loss

    @debug
    def recv_act(self):
        tensor = torch.empty(self.act_shape).bfloat16().cuda()
        dist.recv(tensor, self.rank - 1)
        self.activations.append(tensor)

    @debug
    def send_act(self, act):
        dist.recv(act, self.rank + 1)

    @debug
    def send_grad(self, grad):
        dist.send(grad, self.rank - 1)

    @debug
    def recv_grad(self):
        tensor = torch.empty(self.act_shape).bfloat16().cuda()
        dist.recv(tensor, self.rank + 1)
        return tensor

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def update(self):
        tensor_to_grad(self.buffers["grad"].send / self.world_size, self.model_fp32)
        self.buffers["grad"].send.zero_()
        nn.utils.clip_grad_norm_(self.model_fp32.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.models[0] = copy.deepcopy(self.model_fp32).bfloat16()
