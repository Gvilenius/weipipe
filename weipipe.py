import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import copy
from pprint import pprint
from model import Layer, ModelArgs, Transformer, RMSNorm
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


def num_params(model):
    return sum(x.numel() for x in params(model))


def copy_gradients(model_src, model_dest):
    for param_src, param_dest in zip(model_src.parameters(), model_dest.parameters()):
        if param_dest.grad is None:
            param_dest.grad = torch.zeros_like(param_dest.data)
        param_dest.grad.data.copy_(param_src.grad.data)

    model_src.zero_grad()


def copy_weights(model_src, model_dest):
    for param_src, param_dest in zip(model_src.parameters(), model_dest.parameters()):
        param_dest.data.copy_(param_src.data)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.decoders = Layer(config)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        self.embedding.weight = self.output.weight


class WeiPipe:
    def __init__(self, config, batch_size, gradient_accumulation_steps=1):
        # Setup world info

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        config.n_layers //= self.world_size
        self.config = config

        self.model_32 = Model(config).cuda()
        self.model_16 = Model(config).cuda().bfloat16()

        # forward backup
        self.decoders = Layer(self.config).cuda().bfloat16()

        copy_weights(self.model_32.output, self.model_16.output)
        copy_weights(self.model_32.norm, self.model_16.norm)

        num_decoders_params = num_params(self.decoders)

        self.buffers = {
            "weight0": Buffer(num_decoders_params),
            "weight1": Buffer(num_decoders_params),
            "grad": Buffer(num_decoders_params),
        }

        self.flatten_weight()

        self.current_model_index = 0

        self.loss_fn = loss_fn
        self.activations = ActivationBuffer()

        self.optimizer = configure_optimizers(self.model_32)
        self.optimizer.zero_grad()

        self.counter = {}
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def flatten_weight(self):
        i = 0
        for p in params(self.model_32.decoders):
            n = p.data.numel()
            self.buffers["weight0"].recv[i : i + n] = p.data.view(-1)
            i += n

        def flatten(model, tensor):
            i = 0
            for p in params(model):
                n = p.data.numel()
                p.data = tensor[i : i + n].view(p.data.shape)
                i += n

        flatten(self.model_16.decoders, self.buffers[f"weight{0}"].recv)
        flatten(self.decoders, self.buffers[f"weight{1}"].recv)

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
        dst_rank = (self.world_size + 1 - self.rank) % self.world_size
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
        grad_to_tensor(self.model_16.decoders, grad_buffer.send)
        reqs = dist.batch_isend_irecv(self.flow_op("grad"))
        self.buffers["grad"].pingpong()
        return reqs

    def forward(self, x):
        return self.get_full_transformer(x)

    def forward_step(self, i_layer):
        x = self.activations.top()
        with torch.no_grad():
            x = self.decoders(x)
        self.activations.push(x)

    def backward_step(self, grad=None, i_layer=-1):
        inputs = self.activations.pop().detach()
        inputs.requires_grad = True

        # recomputation
        outputs = self.model_16.decoders(inputs)
        outputs.backward(grad)

        return inputs.grad

    def preprocess(self, x):
        x = self.model_16.embedding(x)
        x = self.model_16.dropout(x)
        return x

    def postprocess(self, x):
        x = self.model_16.norm(x)
        x = self.model_16.output(x)
        return x

    def calc_grad(self, targets=None):
        outputs = self.activations.pop()

        outputs.requires_grad = True
        loss = self.loss_fn(self.postprocess(outputs), targets)
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

        embedding_x = []
        embedding_grad = []
        x = self.preprocess(inputs[0])
        self.activations.push(x)
        embedding_x.append(x)

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

            x = self.preprocess(inputs[i + 1])
            self.activations.push(x)
            embedding_x.append(x)

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
            embedding_grad.append(grad)

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
                if i == self.rank:
                    embedding_grad.append(grad)

            b_reqs = self.backward_weight_flow()

            wait(g_reqs)
            g_reqs = self.grad_flow()

        for i in range(gradient_accumulation_steps):
            embedding_x[i].backward(embedding_grad[i])

        wait(f_reqs)
        wait(b_reqs)
        wait(g_reqs)

        self.activations.reverse()
        self.update()
        return loss

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    # def get_full_transformer(self):
    #     n = sum(x.numel() for x in self.layer_fp32.layers.parameters())
    #     n_embedding = sum(
    #         x.numel() for x in self.layer_fp32.tok_embeddings.parameters()
    #     )
    #     tensor = init_tensor(n, dtype=torch.float32)
    #     model_to_tensor(self.layer_fp32.layers, tensor)

    #     sector_len = len(tensor)
    #     transformer = None

    #     if self.rank == 0:
    #         transformer = Transformer(self.config).cuda()
    #         if self.world_size == 1:
    #             transformer.layers = self.layer_fp32.layers
    #             transformer.norm = self.layer_fp32.norm
    #             transformer.output = self.layer_fp32.output

    #             tensor = init_tensor(n_embedding, dtype=torch.float32)
    #             model_to_tensor(self.layer_fp32.tok_embeddings, tensor)
    #             tensor_to_model(tensor, transformer.tok_embeddings)
    #             # transformer.tok_embeddings = self.layer_fp32.tok_embeddings
    #         else:
    #             tensors = [
    #                 torch.empty(sector_len).float().cuda()
    #                 for i in range(self.world_size)
    #             ]

    #             transformer.norm = self.layer_fp32.norm
    #             transformer.output = self.layer_fp32.output

    #             dist.gather(tensor, tensors)
    #             tensors = tensors[1:] + [tensor]

    #             tensor_to_model(torch.hstack(tensors), transformer.layers)

    #             tensor = init_tensor(n_embedding, dtype=torch.float32)
    #             dist.recv(tensor, 1)
    #             tensor_to_model(tensor, transformer.tok_embeddings)

    #     else:
    #         dist.gather(tensor)
    #         if self.rank == 1:
    #             tensor = init_tensor(n_embedding, dtype=torch.float32)
    #             model_to_tensor(self.layer_fp32.tok_embeddings, tensor)
    #             dist.send(
    #                 tensor,
    #                 0,
    #             )

    #     dist.barrier()
    #     return transformer

    def update(self):
        # exchange params for embedding
        num_params_embedding = num_params(self.model_16.embedding)
        num_params_norm = num_params(self.model_16.norm)
        grad_buffer = init_tensor(num_params_embedding + num_params_norm)
        grad_to_tensor(self.model_16.embedding, grad_buffer[0:num_params_embedding])
        grad_to_tensor(self.model_16.norm, grad_buffer[num_params_embedding:])

        dist.all_reduce(grad_buffer)

        grad_buffer /= self.world_size * self.gradient_accumulation_steps

        tensor_to_grad(grad_buffer[0:num_params_embedding], self.model_32.embedding)
        tensor_to_grad(grad_buffer[0:num_params_embedding], self.model_32.output)
        tensor_to_grad(grad_buffer[num_params_embedding:], self.model_32.norm)

        self.buffers["grad"].send /= self.world_size * self.gradient_accumulation_steps

        tensor_to_grad(
            self.buffers["grad"].send,
            self.model_32.decoders,
        )
        self.buffers["grad"].send.zero_()
        self.model_16.decoders.zero_grad()

        nn.utils.clip_grad_norm_(params(self.model_32), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        copy_weights(self.model_32.output, self.model_16.output)
        copy_weights(self.model_32.norm, self.model_16.norm)

        i = 0
        for p in params(self.model_32.decoders):
            n = p.data.numel()
            self.buffers["weight0"].recv[i : i + n] = p.data.view(-1).bfloat16()
            i += n
