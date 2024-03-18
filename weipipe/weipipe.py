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
        print(self.rank, func.__name__)
        return func(self, *args)

    return wrapper


def wait(reqs):
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


class WeiPipe:
    def __init__(self, config, batch_size):
        # Setup world info

        self.config = config
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.model_fp32 = Layer(self.rank, self.world_size, self.config).float().cuda()

        self.models = [
            copy.deepcopy(self.model_fp32).bfloat16(),
            Layer(self.rank, self.world_size, self.config).bfloat16().cuda(),
        ]

        self.n_parameter = sum(x.numel() for x in self.models[0].parameters())

        self.buffers = {
            "weight0": Buffer(self.n_parameter),
            "weight1": Buffer(self.n_parameter),
            "grad": Buffer(self.n_parameter),
        }

        self.current_model_index = 0

        self.loss_fn = loss_fn
        self.activations = []

        self.optimizer = configure_optimizers(self.model_fp32)
        self.optimizer.zero_grad()

        self.flattern_weight()

    def flattern_weight(self):
        i = 0
        for p in self.models[0].parameters():
            n = p.data.numel()
            self.buffers["weight0"].recv[i : i + n] = p.data.view(-1)
            i += n

        for j in range(2):
            i = 0
            for n, p in self.models[j].named_parameters():
                n = p.data.numel()
                p.data = self.buffers[f"weight{j}"].recv[i : i + n].view(p.data.shape)
                i += n

    def weight_swap(self):
        """At the begining, swap weight between rank i and rank n-i"""
        dst_rank = self.world_size - 1 - self.rank

        weight_buffer = self.buffers["weight0"]
        weight_buffer.send.copy_(weight_buffer.recv)

        send_op = dist.P2POp(dist.isend, weight_buffer.send, dst_rank)
        recv_op = dist.P2POp(dist.irecv, self.buffers["weight1"].recv, dst_rank)

        reqs = dist.batch_isend_irecv([send_op, recv_op])

        for r in reqs:
            r.wait()

    def flow_op(self, idx):
        prev_rank = (self.rank + self.world_size - 1) % self.world_size
        next_rank = (self.rank + 1) % self.world_size
        buffer = self.buffers[idx]
        send_op = dist.P2POp(dist.isend, buffer.send, next_rank)
        recv_op = dist.P2POp(dist.irecv, buffer.recv, prev_rank)
        return [send_op, recv_op]

    def weight_grad_flow(self):
        for i in range(2):
            # model_to_tensor(self.models[i], self.buffers[f"weight{i}"].send)
            weight_buffer = self.buffers[f"weight{i}"]
            weight_buffer.send.copy_(weight_buffer.recv)

        grad_flow_op = self.flow_op("grad")
        weight_flow_op = self.flow_op("weight0") + self.flow_op("weight1")
        reqs = dist.batch_isend_irecv(grad_flow_op + weight_flow_op)
        for req in reqs:
            req.wait()
        self.buffers["grad"].pingpong()

    def forward_model(self):
        self.current_model_index = 1
        return self.models[self.current_model_index]

    def backward_model(self):
        self.current_model_index = 0
        return self.models[self.current_model_index]

    def print_model(self, all=False):
        model_str = str(*self.current_model().parameters())
        if self.current_model_index == 0:
            msg = f"rank{self.rank} forward using {model_str}"
        else:
            msg = f"rank{self.rank} backward using {model_str}"

        if all:
            model_str0 = str(*self.models[0].parameters())
            model_str1 = str(*self.models[1].parameters())
            msg = f"rank{self.rank} all model parameter is {model_str0} {model_str1}"

        print(msg)

    def forward(self, x):
        return self.get_full_transformer(x)

    def forward_step(self, is_first, is_last):
        self.forward_model()
        x = self.activations[-1]
        with torch.no_grad():
            y = self.current_model()(x, is_first=is_first, is_last=is_last)
        self.activations.append(y)

    def backward_step(self, grad, is_first, is_last):
        self.backward_model()
        inputs = self.activations.pop().detach()
        if not is_first:
            inputs.requires_grad = True
        # recomputation
        outputs = self.current_model()(inputs, is_first=is_first, is_last=is_last)
        outputs.backward(grad)

        grad_buffer = self.buffers["grad"]

        grad_to_tensor(self.current_model(), grad_buffer.send)
        return inputs.grad

    def calc_grad(self, targets):
        outputs = self.activations.pop()
        outputs.requires_grad = True
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        grad = outputs.grad
        return grad, loss

    def forward_backward_step(self, inputs, targets=None):
        bsz, seq_len = inputs.shape

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        self.weight_swap()
        # print(prof.key_averages().table(sort_by="cuda_time"))

        # warmup
        for i in range(self.rank):
            self.weight_grad_flow()

        self.activations.append(inputs)

        for i in range(self.world_size):
            self.forward_step(i == 0, i == self.world_size - 1)
            self.weight_grad_flow()

        grad, loss = self.calc_grad(targets)

        for i in range(self.world_size):
            grad = self.backward_step(grad, i == self.world_size - 1, i == 0)
            self.weight_grad_flow()

        # cooldown
        for i in range(self.world_size - self.rank):
            self.weight_grad_flow()
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
        tensor_to_grad(self.buffers["grad"].send / self.world_size, self.model_fp32)
        self.buffers["grad"].send.zero_()
        nn.utils.clip_grad_norm_(self.model_fp32.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        i = 0
        for p in self.model_fp32.parameters():
            n = p.data.numel()
            self.buffers["weight0"].recv[i : i + n] = p.data.view(-1).bfloat16()
            i += n

    def current_model(self):
        return self.models[self.current_model_index]


class WeiPipeAccum:
    def __init__(self, config, batch_size):
        # Setup world info

        self.config = config
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.model_fp32 = Layer(self.rank, self.world_size, self.config).float().cuda()

        self.models = [
            copy.deepcopy(self.model_fp32).bfloat16(),
            Layer(self.rank, self.world_size, self.config).bfloat16().cuda(),
        ]

        self.n_parameter = sum(x.numel() for x in self.models[0].parameters())

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
        self.gradient_accumulation_steps = 4

    def flattern_weight(self):
        i = 0
        for p in self.models[0].parameters():
            n = p.data.numel()
            self.buffers["weight0"].recv[i : i + n] = p.data.view(-1)
            i += n

        for j in range(2):
            i = 0
            for n, p in self.models[j].named_parameters():
                n = p.data.numel()
                p.data = self.buffers[f"weight{j}"].recv[i : i + n].view(p.data.shape)
                i += n

    def weight_swap(self):
        """At the begining, swap weight between rank i and rank n-i"""
        dst_rank = self.world_size - 1 - self.rank
        send_op = dist.P2POp(dist.isend, self.buffers["weight0"].recv, dst_rank)
        recv_op = dist.P2POp(dist.irecv, self.buffers["weight1"].recv, dst_rank)
        reqs = dist.batch_isend_irecv([send_op, recv_op])
        return reqs

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
        # print(idx, weight_buffer.buffers)
        return dist.batch_isend_irecv(weight_flow_op)

    def forward_weight_flow(self):
        return self.weight_flow(1)

    def backward_weight_flow(self):
        return self.weight_flow(0)

    def grad_flow(self):
        wait(dist.batch_isend_irecv(self.flow_op("grad")))
        self.buffers["grad"].pingpong()

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
        grad_buffer = self.buffers["grad"]
        grad_to_tensor(self.backward_model(), grad_buffer.send)

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
        for _ in range(self.rank):
            wait(self.forward_weight_flow())

        for i in range(self.world_size - self.rank):
            self.forward_step(i_layer=i)
            wait(self.forward_weight_flow())

        for i in range(self.rank):
            wait(self.backward_weight_flow())
            self.forward_step(i_layer=i)
            wait(self.forward_weight_flow())

        for i in range(gradient_accumulation_steps):
            self.activations.reverse()
            self.activations.push(inputs[i + 1])

            grad, loss = self.calc_grad(targets[i])
            for j in range(self.world_size):
                grad = self.backward_step(
                    grad=grad,
                    i_layer=self.world_size - 1 - j,
                )
                self.grad_flow()
                wait(self.backward_weight_flow())

                self.forward_step(i_layer=j)
                wait(self.forward_weight_flow())

        grad, loss = self.calc_grad(targets[-1])
        self.activations.reverse()
        for i in range(self.world_size - 1 - self.rank):
            grad = self.backward_step(grad=grad, i_layer=self.world_size - 1 - i)
            self.grad_flow()
            wait(self.backward_weight_flow())
            wait(self.forward_weight_flow())

        for i in range(self.world_size):
            if i <= self.rank:
                grad = self.backward_step(grad=grad, i_layer=self.world_size - 1 - i)
            self.grad_flow()
            wait(self.backward_weight_flow())

        self.update()
        exit()
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

    def print_weight(self):
        def pp(m):
            s = ""
            for p in m.parameters():
                s += str(p.data) + "\n"
            print(s)

        for i in range(0, self.world_size - 1):
            dist.barrier()
            if self.rank == i + 1:
                pp(self.model_fp32)
        dist.barrier()
        if self.rank == 0:
            pp(self.model_fp32)

    def update(self):
        tensor_to_grad(self.buffers["grad"].send / self.world_size, self.model_fp32)
        self.buffers["grad"].send.zero_()
        nn.utils.clip_grad_norm_(self.model_fp32.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        i = 0
        for p in self.model_fp32.parameters():
            n = p.data.numel()
            self.buffers["weight0"].recv[i : i + n] = p.data.view(-1).bfloat16()
            i += n
