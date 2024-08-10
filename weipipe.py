import torch
import torch.distributed
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import copy
from pprint import pprint
from model import Layer, ModelArgs, Transformer, RMSNorm
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext
import queue
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

    def push(self, x, detach=False):
        if detach:
            x = x.detach()
            x.requires_grad = True

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


def flatten(model, tensor):
    i = 0
    for p in params(model):
        n = p.data.numel()
        p.data = tensor[i : i + n].view(p.data.shape)
        i += n


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
    def __init__(self, config, gradient_accumulation_steps=1, train_embedding=False):
        # Setup world info

        self.train_embedding = train_embedding
        self.enable_checkpointing = config.checkpointing
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.config = copy.deepcopy(config)

        config.n_layers //= self.world_size

        if train_embedding:
            self.model_32 = Model(config).cuda()
        else:
            self.model_32 = Layer(config).cuda()
            
        self.model_16 = Model(config).cuda().half()
        # forward backup
        self.decoders = Layer(config).cuda().half()

        if train_embedding:
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

        # trainable_modules = nn.ModuleList ([self.model_32.decoders, self.model_32.norm])
        trainable_modules = self.model_32
        self.optimizer = configure_optimizers(trainable_modules)

        self.optimizer.zero_grad()

        self.counter = {}
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def flatten_weight(self):
        i = 0
        for p in params(self.model_32.decoders):
            n = p.data.numel()
            self.buffers["weight0"].recv[i : i + n] = p.data.view(-1)
            i += n
  
        flatten(self.model_16.decoders, self.buffers[f"weight{0}"].recv)
        flatten(self.decoders, self.buffers[f"weight{1}"].recv)

    # ring exchange
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

        if self.enable_checkpointing:
            grad_to_tensor(self.decoders, grad_buffer.send)
        else:
            grad_to_tensor(self.model_16.decoders, grad_buffer.send)

        reqs = dist.batch_isend_irecv(self.flow_op("grad"))
        self.buffers["grad"].pingpong()
        return reqs

    def forward(self, x):
        return self.get_full_transformer(x)

    def forward_step(self):
        x = self.activations.top()
        x.retain_grad()

        ctx = nullcontext() if self.enable_checkpointing else torch.no_grad()

        with ctx:
            x = self.decoders(x)

        self.activations.push(x)
        self.activations.push(x, detach=True)

    def backward_step(self, grad=None):
        outputs = self.activations.pop()
        inputs = self.activations.pop()

        # recomputation
        if not self.enable_checkpointing:
            outputs = self.model_16.decoders(inputs)
            outputs.backward(grad)
        else:
            # replace weight and then do checkpointing
            flatten(self.decoders, self.buffers[f"weight{0}"].recv)
            outputs.backward(grad)
            flatten(self.decoders, self.buffers[f"weight{1}"].recv)

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

        loss = self.loss_fn(self.postprocess(outputs), targets)
        loss.backward()
        grad = outputs.grad
        return grad, loss

    def forward_backward_step(self, dl_iter):
        wait(self.weight_swap())

        X, Y = next(dl_iter)
        x = self.preprocess(X)
        self.activations.push(x, detach=True)

        f_reqs = None
        b_reqs = None
        g_reqs = None

        for _ in range(self.rank):
            wait(f_reqs)
            f_reqs = self.forward_weight_flow()

        for i in range(self.world_size - self.rank):
            wait(f_reqs)
            self.forward_step()
            f_reqs = self.forward_weight_flow()

        for i in range(self.rank):
            wait(b_reqs)
            b_reqs = self.backward_weight_flow()

            wait(g_reqs)
            g_reqs = self.grad_flow()

            wait(f_reqs)
            self.forward_step()
            f_reqs = self.forward_weight_flow()

        x1 = x
        for i in range(self.gradient_accumulation_steps - 1):

            self.activations.reverse()

            grad, loss = self.calc_grad(Y)

            X, Y = next(dl_iter)

            x1 = self.preprocess(X)
            self.activations.push(x1, detach=True)

            for _ in range(self.world_size):
                wait(b_reqs)
                grad = self.backward_step(grad=grad)
                b_reqs = self.backward_weight_flow()

                wait(g_reqs)
                g_reqs = self.grad_flow()

                wait(f_reqs)
                self.forward_step()
                f_reqs = self.forward_weight_flow()

            if i == self.gradient_accumulation_steps - 2 and  self.train_embedding:
                x.backward(grad)
                
        x = x1
        self.activations.reverse()

        grad, loss = self.calc_grad(Y)

        for i in range(self.world_size - 1 - self.rank):
            wait(b_reqs)
            grad = self.backward_step(grad=grad)
            b_reqs = self.backward_weight_flow()

            wait(g_reqs)
            g_reqs = self.grad_flow()

            wait(f_reqs)
            f_reqs = self.forward_weight_flow()

        for i in range(self.world_size+1):
            wait(b_reqs)
            if i <= self.rank:
                grad = self.backward_step(grad=grad)

                if i == self.rank and self.train_embedding:
                    x.backward(grad)

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
        transformer = None
        num_decoders_params = num_params(self.decoders)

        tensor = init_tensor(num_decoders_params, dtype=torch.float)

        def flatten(model, tensor):
            i = 0
            for p in model.parameters():
                n = p.data.numel()
                tensor[i : i + n] = p.data.view(-1)
                i += n

        def unflatten(tensor, model):
            i = 0
            for p in model.parameters():
                n = p.data.numel()
                p.data = tensor[i : i + n].reshape(p.data.shape)
                i += n

        flatten(self.model_32.decoders, tensor)

        if self.rank == 0:
            transformer = Transformer(self.config).cuda()
            transformer.norm = self.model_32.norm
            transformer.output = self.model_32.output
            transformer.tok_embeddings = self.model_32.embedding

            if self.world_size == 1:
                transformer.layers = self.model_32.decoders
            else:
                tensors = [
                    init_tensor(num_decoders_params, dtype=torch.float)
                    for i in range(self.world_size)
                ]
                dist.gather(tensor, tensors)
                tensors = tensors[1:] + [tensor]
                unflatten(torch.hstack(tensors), transformer.layers)
        else:
            dist.gather(tensor)

        return transformer

    def update(self):
        # exchange params for embedding
    
        if self.train_embedding:
            num_params_embedding = num_params(self.model_16.embedding)
            num_params_norm = num_params(self.model_16.norm)
            grad_buffer = init_tensor(num_params_embedding + num_params_norm)

            grad_to_tensor(self.model_16.embedding, grad_buffer[0:num_params_embedding])
            grad_to_tensor(self.model_16.norm, grad_buffer[num_params_embedding:])
            dist.all_reduce(grad_buffer, async_op=False)

            grad_buffer /= self.world_size * self.gradient_accumulation_steps
            tensor_to_grad(grad_buffer[0:num_params_embedding], self.model_32.output)
            tensor_to_grad(grad_buffer[num_params_embedding:], self.model_32.norm)


        self.buffers["grad"].send /= self.world_size * self.gradient_accumulation_steps
        tensor_to_grad(
            self.buffers["grad"].send,
            self.model_32.decoders,
        )
        self.buffers["grad"].send.zero_()
        # nn.utils.clip_grad_norm_(params(self.model_32), 1.0)

        self.optimizer.step()
        self.optimizer.zero_grad()

        # copy model32 to model16

        if self.train_embedding:
            copy_weights(self.model_32.output, self.model_16.output)
            copy_weights(self.model_32.norm, self.model_16.norm)

        i = 0
        for p in params(self.model_32.decoders):
            n = p.data.numel()
            self.buffers["weight0"].recv[i : i + n] = p.data.view(-1).half()
            i += n

