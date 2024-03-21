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


embedding = False


# broadcast version
class FSDP_broadcast:
    def __init__(self, config, batch_size, gradient_accumulation_steps=1):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.config = config
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.model_32 = Layer(self.rank, self.world_size, self.config).float().cuda()
        self.model_16 = Layer(self.rank, self.world_size, self.config).bfloat16().cuda()

        self.working_model = (
            Layer(self.rank, self.world_size, self.config).bfloat16().cuda()
        )

        self.n_parameter = sum(
            x.numel() for x in params(self.model_16, embedding=embedding)
        )

        self.working_model_flattened = init_tensor(self.n_parameter)
        self.model_16_flattened = init_tensor(self.n_parameter)

        self.loss_fn = loss_fn

        self.optimizer = configure_optimizers(self.model_32)
        self.optimizer.zero_grad()

        self.flatten_weight()

    def acquire_weight(self, rank):
        if rank == self.rank:
            self.working_model_flattened.copy_(self.model_16_flattened)

        dist.broadcast(self.working_model_flattened, rank)

    def _update_16(self):
        i = 0
        for p in params(self.model_32):
            n = p.data.numel()
            self.model_16_flattened[i : i + n] = p.data.view(-1).bfloat16()
            i += n

    def flatten_weight(self):
        i = 0
        for p in params(self.model_32):
            n = p.data.numel()
            self.model_16_flattened[i : i + n] = p.data.view(-1).bfloat16()
            i += n

        i = 0
        for p in params(self.model_16):
            n = p.data.numel()
            p.data = self.model_16_flattened[i : i + n].view(p.data.shape)
            i += n

        i = 0
        for p in params(self.working_model):
            n = p.data.numel()
            p.data = self.working_model_flattened[i : i + n].view(p.data.shape)
            i += n

    # mark

    def forward(self, x, i_layer):
        return self.working_model(x, i_layer == 0, i_layer == self.world_size - 1)

    def forward_backward_step(self, inputs, targets=None):
        gradient_accumulation_steps = self.gradient_accumulation_steps
        bsz, seq_len = inputs.shape
        micro_bsz = bsz // gradient_accumulation_steps

        inputs = torch.split(inputs, micro_bsz, 0)
        targets = torch.split(targets, micro_bsz, 0)

        # FSDP 做梯度累加通信代价是很大的,待验证
        grad_buffer = init_tensor(self.n_parameter)
        for i in range(gradient_accumulation_steps):
            activations = [inputs[i]]
            y = targets[i]

            # partition number
            i_layer = 0

            # forward
            for i_layer in range(self.world_size - 1):
                self.acquire_weight(i_layer)
                with torch.no_grad():
                    act = self.forward(activations[-1], i_layer=i_layer)
                activations.append(act)

            # backward
            for i_layer in reversed(range(self.world_size)):
                self.acquire_weight(i_layer)
                x = activations.pop().detach()
                if i_layer != 0:
                    x.requires_grad = True
                y_ = self.forward(x, i_layer=i_layer)
                if i_layer == self.world_size - 1:
                    loss = self.loss_fn(y_, y)
                    loss.backward()
                    grad = x.grad
                else:
                    y_.backward(grad)
                    grad = x.grad
                grad_to_tensor(self.working_model, grad_buffer)
                dist.reduce(grad_buffer, i_layer)
                if self.rank == i_layer:
                    tensor_to_grad(grad_buffer, self.model_32)

        self.update()
        return loss

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def update(self):
        nn.utils.clip_grad_norm_(params(self.model_32), 1.0)

        self.optimizer.step()
        self.optimizer.zero_grad()

        self._update_16()


class FSDP_allgather:
    def __init__(self, config, batch_size, gradient_accumulation_steps=1):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.config = config
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.model_32 = Layer(self.rank, self.world_size, self.config).float().cuda()
        self.model_16 = Layer(self.rank, self.world_size, self.config).bfloat16().cuda()

        self.config.n_layers *= self.world_size
        self.working_model = (
            Layer(self.rank, self.world_size, self.config).bfloat16().cuda()
        )
        self.config.n_layers //= self.world_size

        self.n_parameter = sum(x.numel() for x in params(self.model_16))

        self.working_model_flattened = [init_tensor(self.n_parameter)] * self.world_size
        self.model_16_flattened = init_tensor(self.n_parameter)

        self.loss_fn = loss_fn

        self.optimizer = configure_optimizers(self.model_32)
        self.optimizer.zero_grad()

        self.flatten_weight()

    def acquire_weight(self):
        dist.all_gather(self.working_model_flattened, self.model_16_flattened)

    def _update_16(self):
        i = 0
        for p in params(self.model_32):
            n = p.data.numel()
            self.model_16_flattened[i : i + n] = p.data.view(-1).bfloat16()
            i += n

    def flatten_weight(self):
        i = 0
        for p in params(self.model_32, embedding=embedding):
            n = p.data.numel()
            self.model_16_flattened[i : i + n] = p.data.view(-1).bfloat16()
            i += n

        i = 0
        for p in params(self.model_16, embedding=embedding):
            n = p.data.numel()
            p.data = self.model_16_flattened[i : i + n].view(p.data.shape)
            i += n

        i = 0
        for p in self.working_model.layers.parameters():
            n = p.data.numel()

            j = i // self.n_parameter

            start = i % self.n_parameter
            end = (i + n) % self.n_parameter

            p.data = self.working_model_flattened[j][start:end].view(p.data.shape)
            i += n

    # mark

    def forward(self, x, i_layer):
        return self.working_model(x, i_layer == 0, i_layer == self.world_size - 1)

    def forward_backward_step(self, inputs, targets=None):
        gradient_accumulation_steps = self.gradient_accumulation_steps
        bsz, seq_len = inputs.shape
        micro_bsz = bsz // gradient_accumulation_steps

        inputs = torch.split(inputs, micro_bsz, 0)
        targets = torch.split(targets, micro_bsz, 0)

        # FSDP 做梯度累加通信代价是很大的,待验证
        grad_buffer = init_tensor(self.n_parameter)

        for i in range(gradient_accumulation_steps):
            x = inputs[i]
            y = targets[i]

            self.acquire_weight()
            y_ = self.working_model(x, True, True)
            loss = self.loss_fn(y_, y)
            self.acquire_weight()
            loss.backward()
            grad_buffer = init_tensor(self.n_parameter * self.world_size)
            grad_to_tensor(self.working_model.layers, grad_buffer)

        dist.all_reduce(grad_buffer)
        start = self.rank * self.n_parameter
        end = start + self.n_parameter
        grad_buffer = grad_buffer[start:end]
        tensor_to_grad(grad_buffer, self.model_32)

        self.update()
        return loss

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def update(self):
        nn.utils.clip_grad_norm_(params(self.model_32), 1.0)

        self.optimizer.step()
        self.optimizer.zero_grad()

        self._update_16()
