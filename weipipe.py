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
    model_to_tensor,
    tensor_to_model,
    init_tensor,
    print_rank,
)


class ActivationBuffer:
    def __init__(self):
        self.activations = []
        self._reverse = False

    def reverse(self):
        self._reverse = not self._reverse

    def push(self, x):
        if not self._reverse:
            self.activations.append(x)
        else:
            self.activations.insert(0, x)

    # for backward
    def pop(self):
        if not self._reverse:
            y = self.activations[0]
            del self.activations[0]
            return y
        else:
            return self.activations.pop()

    # for forward
    def top(self):
        if not self._reverse:
            return self.activations[-1]
        else:
            return self.activations[0]


def debug(func):
    def wrapper(self, *args):
        if func.__name__ not in self.counter:
            self.counter[func.__name__] = 0
        self.counter[func.__name__] += 1
        return func(self, *args)

    return wrapper


def wait(reqs):
    if reqs is not None:
        for r in reqs:
            r.wait()


class Buffer:
    def __init__(self, n):
        self.buffers = [
            init_tensor(n, init_func=torch.zeros),
            init_tensor(n, init_func=torch.zeros),
        ]

        self.index = 0
        self.send = self.buffers[0]
        self.recv = self.buffers[1]

    def pingpong(self):
        self.index = 1 - self.index
        self.send = self.buffers[self.index]
        self.recv = self.buffers[1 - self.index]


def params(m):
    return m.parameters()
    # return m.layers.parameters()


class WeiPipe:
    def __init__(self, config, batch_size, gradient_accumulation_steps=1):
        # Setup world info

        self.config = config
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.model_fp32 = Layer(self.rank, self.world_size, self.config).float().cuda()

        self.models = [
            copy.deepcopy(self.model_fp32).bfloat16(),
            Layer(self.rank, self.world_size, self.config).bfloat16().cuda(),
        ]

        self.n_parameter = sum(x.numel() for x in params(self.models[0]))

        self.buffers = {
            "weight0": Buffer(self.n_parameter),
            "weight1": Buffer(self.n_parameter),
            "grad": Buffer(self.n_parameter),
        }

        self.current_model_index = 0

        self.loss_fn = loss_fn
        self.activations = ActivationBuffer()

        self.optimizer = configure_optimizers(self.model_fp32)
        self.optimizer.zero_grad()

        self.flattern_weight()
        self.counter = {}
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def flattern_weight(self):
        i = 0
        for p in params(self.models[0]):
            n = p.data.numel()
            self.buffers["weight0"].recv[i : i + n] = p.data.view(-1)
            i += n

        for j in range(2):
            i = 0
            for p in params(self.models[j]):
                n = p.data.numel()
                p.data = self.buffers[f"weight{j}"].recv[i : i + n].view(p.data.shape)
                i += n

    def flow_op(self, idx):
        prev_rank = (self.rank + self.world_size - 1) % self.world_size
        next_rank = (self.rank + 1) % self.world_size
        buffer = self.buffers[idx]
        send_op = dist.P2POp(dist.isend, buffer.send, next_rank)
        recv_op = dist.P2POp(dist.irecv, buffer.recv, prev_rank)
        return [send_op, recv_op]

    def weight_flow(self, idx):
        weight_buffer = self.buffers[f"weight{idx}"]
        weight_buffer.send.copy_(weight_buffer.recv)
        weight_flow_op = self.flow_op(f"weight{idx}")
        return dist.batch_isend_irecv(weight_flow_op)

    def weight_swap(self):
        """At the begining, swap weight between rank i and rank n-i"""
        dst_rank = self.world_size - 1 - self.rank
        send_op = dist.P2POp(dist.isend, self.buffers["weight0"].recv, dst_rank)
        recv_op = dist.P2POp(dist.irecv, self.buffers["weight1"].recv, dst_rank)
        reqs = dist.batch_isend_irecv([send_op, recv_op])
        return reqs

    def forward_weight_flow(self):
        return self.weight_flow(1)

    def backward_weight_flow(self):
        return self.weight_flow(0)

    def grad_flow(self):
        grad_buffer = self.buffers["grad"]
        grad_to_tensor(self.backward_model(), grad_buffer.send)
        reqs = dist.batch_isend_irecv(self.flow_op("grad"))
        self.buffers["grad"].pingpong()
        return reqs

    def forward_model(self):
        self.current_model_index = 1
        return self.models[self.current_model_index]

    def backward_model(self):
        self.current_model_index = 0
        return self.models[self.current_model_index]

    def forward(self, x):
        return self.get_full_transformer(x)

    def forward_step(self, i_layer):
        x = self.activations.top()
        with torch.no_grad():
            y = self.forward_model()(
                x, is_first=i_layer == 0, is_last=i_layer == self.world_size - 1
            )
        self.activations.push(y)

    def backward_step(self, grad=None, i_layer=-1):
        is_first = i_layer == 0
        is_last = i_layer == self.world_size - 1

        inputs = self.activations.pop().detach()
        if not is_first:
            inputs.requires_grad = True
        # recomputation
        outputs = self.backward_model()(inputs, is_first=is_first, is_last=is_last)
        outputs.backward(grad)

        return inputs.grad

    def calc_grad(self, targets=None):
        outputs = self.activations.pop()

        outputs.requires_grad = True
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        grad = outputs.grad
        return grad, loss

    # mark
    def forward_backward_step(self, inputs, targets=None):
        ### fuck async

        gradient_accumulation_steps = self.gradient_accumulation_steps
        bsz, seq_len = inputs.shape
        micro_bsz = bsz // gradient_accumulation_steps

        inputs = torch.split(inputs, micro_bsz, 0)
        targets = torch.split(targets, micro_bsz, 0)
        wait(self.weight_swap())
        # ------------------------------

        self.activations.push(inputs[0])
        f_reqs = None
        b_reqs = None
        g_reqs = None
        for _ in range(self.rank):
            wait(f_reqs)
            f_reqs = self.forward_weight_flow()

        for i in range(self.world_size - self.rank):
            wait(f_reqs)
            self.forward_step(i_layer=i)
            f_reqs = self.forward_weight_flow()

        for i in range(self.rank):
            wait(b_reqs)
            b_reqs = self.backward_weight_flow()

            wait(g_reqs)
            g_reqs = self.grad_flow()

            wait(f_reqs)
            self.forward_step(i_layer=self.world_size + i - self.rank)
            f_reqs = self.forward_weight_flow()

        for i in range(gradient_accumulation_steps - 1):
            self.activations.reverse()
            self.activations.push(inputs[i + 1])

            grad, loss = self.calc_grad(targets[i])
            for j in range(self.world_size):
                wait(b_reqs)
                grad = self.backward_step(
                    grad=grad,
                    i_layer=self.world_size - 1 - j,
                )
                b_reqs = self.backward_weight_flow()

                wait(g_reqs)
                g_reqs = self.grad_flow()

                wait(f_reqs)
                self.forward_step(i_layer=j)
                f_reqs = self.forward_weight_flow()

        self.activations.reverse()
        grad, loss = self.calc_grad(targets[-1])
        for i in range(self.world_size - 1 - self.rank):
            wait(b_reqs)
            grad = self.backward_step(grad=grad, i_layer=self.world_size - 1 - i)
            b_reqs = self.backward_weight_flow()

            wait(g_reqs)
            g_reqs = self.grad_flow()

            wait(f_reqs)
            f_reqs = self.forward_weight_flow()

        for i in range(self.world_size + 1):
            wait(b_reqs)
            if i <= self.rank:
                grad = self.backward_step(grad=grad, i_layer=self.rank - i)
            b_reqs = self.backward_weight_flow()

            wait(g_reqs)
            g_reqs = self.grad_flow()

        wait(f_reqs)
        wait(b_reqs)
        wait(g_reqs)

        self.activations.reverse()
        self.update()
        return loss

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_full_transformer(self):
        n = sum(x.numel() for x in self.model_fp32.layers.parameters())
        n_embedding = sum(
            x.numel() for x in self.model_fp32.tok_embeddings.parameters()
        )
        tensor = init_tensor(n, dtype=torch.float32)
        model_to_tensor(self.model_fp32.layers, tensor)

        sector_len = len(tensor)
        transformer = None

        if self.rank == 0:
            transformer = Transformer(self.config).cuda()
            if self.world_size == 1:
                transformer.layers = self.model_fp32.layers
                transformer.norm = self.model_fp32.norm
                transformer.output = self.model_fp32.output

                tensor = init_tensor(n_embedding, dtype=torch.float32)
                model_to_tensor(self.model_fp32.tok_embeddings, tensor)
                tensor_to_model(tensor, transformer.tok_embeddings)
                # transformer.tok_embeddings = self.model_fp32.tok_embeddings
            else:
                tensors = [
                    torch.empty(sector_len).float().cuda()
                    for i in range(self.world_size)
                ]

                transformer.norm = self.model_fp32.norm
                transformer.output = self.model_fp32.output

                dist.gather(tensor, tensors)
                tensors = tensors[1:] + [tensor]

                tensor_to_model(torch.hstack(tensors), transformer.layers)

                tensor = init_tensor(n_embedding, dtype=torch.float32)
                dist.recv(tensor, 1)
                tensor_to_model(tensor, transformer.tok_embeddings)

        else:
            dist.gather(tensor)
            if self.rank == 1:
                tensor = init_tensor(n_embedding, dtype=torch.float32)
                model_to_tensor(self.model_fp32.tok_embeddings, tensor)
                dist.send(
                    tensor,
                    0,
                )

        dist.barrier()
        return transformer

    def update(self):
        tensor_to_grad(
            self.buffers["grad"].send
            / self.world_size
            / self.gradient_accumulation_steps,
            self.model_fp32,
        )
        self.buffers["grad"].send.zero_()
        self.backward_model().zero_grad()

        nn.utils.clip_grad_norm_(params(self.model_fp32), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        i = 0
        for p in params(self.model_fp32):
            n = p.data.numel()
            self.buffers["weight0"].recv[i : i + n] = p.data.view(-1).bfloat16()
            i += n
