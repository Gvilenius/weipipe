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
)


class Buffer:
    def __init__(self, n):
        self.buffers = [
            init_tensor(n, init_func=torch.zeros),
            init_tensor(n),
        ]

        self.index = 0
        self.send = self.buffers[0]
        self.recv = self.buffers[1]

    def pingpong(self):
        self.index = 1 - self.index
        self.send = self.buffers[self.index]
        self.recv = self.buffers[self.index]


class WeiPipe:
    def __init__(self) -> None:
        # Setup world info
        model_args = dict(
            dim=288,
            n_heads=6,
            n_kv_heads=None,
            vocab_size=32000,
            multiple_of=32,
            max_seq_len=128,
            dropout=0.0,
            n_layers=6,
        )
        self.config = ModelArgs(**model_args)

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

        self.optimizer = configure_optimizers(self.model_fp32)
        self.optimizer.zero_grad()

    def weight_swap(self):
        """At the begining, swap weight between rank i and rank n-i"""
        dst_rank = self.world_size - 1 - self.rank

        weight_buffer = self.buffers["weight0"]
        model_to_tensor(self.models[0], weight_buffer.send)

        send_op = dist.P2POp(dist.isend, weight_buffer.send, dst_rank)
        recv_op = dist.P2POp(dist.irecv, weight_buffer.recv, dst_rank)

        reqs = dist.batch_isend_irecv([send_op, recv_op])
        for r in reqs:
            r.wait()
        tensor_to_model(weight_buffer.recv, self.models[1])

    def flow_op(self, idx):
        prev_rank = (self.rank + self.world_size - 1) % self.world_size
        next_rank = (self.rank + 1) % self.world_size
        buffer = self.buffers[idx]
        send_op = dist.P2POp(dist.isend, buffer.send, next_rank)
        recv_op = dist.P2POp(dist.irecv, buffer.recv, prev_rank)
        return [send_op, recv_op]

    def weight_grad_flow(self):
        model_to_tensor(self.models[0], self.buffers["weight0"].send)
        model_to_tensor(self.models[1], self.buffers["weight1"].send)

        grad_flow_op = self.flow_op("grad")
        weight_flow_op = self.flow_op("weight0") + self.flow_op("weight1")
        reqs = dist.batch_isend_irecv(grad_flow_op + weight_flow_op)
        for req in reqs:
            req.wait()
        self.buffers["grad"].pingpong()

        tensor_to_model(self.buffers["weight0"].recv, self.models[0])
        tensor_to_model(self.buffers["weight1"].recv, self.models[1])

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

    def forward_backward_step(self, inputs, targets=None, first=False):
        # if first:
        self.weight_swap()
        self.activations = [inputs]
        loss = None
        for i in range(self.world_size * 3):
            i_offset = i - self.rank
            is_first = i_offset in [0, self.world_size * 2 - 1]
            is_last = i_offset in [self.world_size - 1, self.world_size]

            # calculate loss
            if i_offset == self.world_size:
                outputs = self.activations.pop()
                outputs.requires_grad = True
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                grad = outputs.grad

            # forward
            if 0 <= i_offset < self.world_size:
                self.forward_model()
                x = self.activations[-1]
                with torch.no_grad():
                    y = self.current_model()(x, is_first=is_first, is_last=is_last)
                self.activations.append(y)
            # backward
            elif self.world_size <= i_offset < self.world_size * 2:
                self.backward_model()
                inputs = self.activations.pop().detach()
                if not is_first:
                    inputs.requires_grad = True
                # recomputation
                outputs = self.current_model()(
                    inputs, is_first=is_first, is_last=is_last
                )
                outputs.backward(grad)
                grad = inputs.grad

                grad_buffer = self.buffers["grad"]
                grad_to_tensor(self.current_model(), grad_buffer.send)

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
                print(tensor, tensors)
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
        self.models[0] = copy.deepcopy(self.model_fp32).bfloat16()

    def current_model(self):
        return self.models[self.current_model_index]
