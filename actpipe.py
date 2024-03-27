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
    print_rank,
)


def debug(func):
    def wrapper(self, *args):
        print(f"rank {dist.get_rank()} in {func.__name__}")
        func(self, *args)
        print(f"rank {dist.get_rank()} out {func.__name__}")

    return wrapper


class ActPipe:
    def __init__(self, config, batch_size, gradient_accumulation_steps=1):
        # Setup world info
        self.config = config
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.batch_size = batch_size

        self.is_first = self.rank == 0
        self.is_last = self.rank == self.world_size - 1

        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.model_fp32 = Layer(self.rank, self.world_size, self.config).float().cuda()
        self.act_shape = (
            self.batch_size // self.gradient_accumulation_steps,
            self.config.max_seq_len,
            self.config.dim,
        )
        self.model = copy.deepcopy(self.model_fp32).bfloat16()

        self.loss_fn = loss_fn
        self.optimizer = configure_optimizers(self.model_fp32)
        self.activations = []

        self.optimizer.zero_grad()

    def forward_step(self):
        x = self.activations[-1]
        with torch.no_grad():
            y = self.model(x, is_first=self.is_first, is_last=self.is_last)
        self.activations.append(y)

    def backward_step(self, grad):
        inputs = self.activations.pop().detach()
        if not self.is_first:
            inputs.requires_grad = True
        # recomputation
        outputs = self.model(inputs, is_first=self.is_first, is_last=self.is_last)
        outputs.backward(grad)
        return inputs.grad

    def calc_grad(self, targets):
        outputs = self.activations.pop().detach()
        outputs.requires_grad = True
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        return outputs.grad, loss

    def forward_backward_step(self, inputs, targets):
        gradient_accumulation_steps = self.gradient_accumulation_steps
        bsz, seq_len = inputs.shape
        micro_bsz = bsz // gradient_accumulation_steps

        if self.rank == 0:
            dist.send(targets, self.world_size - 1)
        if self.rank == self.world_size - 1:
            dist.recv(targets, 0)

        inputs = torch.split(inputs, micro_bsz, 0)
        targets = torch.split(targets, micro_bsz, 0)

        loss = None
        for istep in range(gradient_accumulation_steps):
            input, target = inputs[istep], targets[istep]

            if self.rank == 0:
                self.activations.append(input)
            else:
                self.recv_act()

            self.forward_step()

            if self.rank != self.world_size - 1:
                self.send_act()

        for istep in range(gradient_accumulation_steps):
            if self.rank != self.world_size - 1:
                grad = self.recv_grad()

            if self.rank == self.world_size - 1:
                target = targets[gradient_accumulation_steps - 1 - istep]
                grad, loss = self.calc_grad(target)

            grad = self.backward_step(grad)

            if self.rank != 0:
                self.send_grad(grad)

        self.update()
        return loss

    def recv_act(self):
        tensor = torch.empty(self.act_shape).bfloat16().cuda()
        dist.recv(tensor, self.rank - 1)
        self.activations.append(tensor)

    def send_act(self):
        act = self.activations.pop()
        dist.send(act, self.rank + 1)

    def send_grad(self, grad):
        dist.send(grad, self.rank - 1)

    def recv_grad(self):
        tensor = torch.empty(self.act_shape).bfloat16().cuda()
        dist.recv(tensor, self.rank + 1)
        return tensor

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def update(self):
        for p16, p32 in zip(self.model.parameters(), self.model_fp32.parameters()):
            if p16.grad is not None:
                p32.grad = copy.deepcopy(p16.grad.float())

        nn.utils.clip_grad_norm_(self.model_fp32.parameters(), 1.0)
        # s = f"------------rank{self.rank}-grad------------\n"
        # for n, p in self.model_fp32.named_parameters():
        #     s += n + str(p.grad) + "\n"
        # print(s)
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.model = copy.deepcopy(self.model_fp32).bfloat16()
