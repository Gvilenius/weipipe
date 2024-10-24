import torch
import torch.distributed
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import copy
from pprint import pprint
from model import Model, Transformer
from contextlib import nullcontext
import numpy as np
from datetime import timedelta as timedelta
import os

from utils import (
    loss_fn,
    configure_optimizers,
    grad_to_tensor,
    tensor_to_grad,
    init_tensor,
    params,
    print_rank,
    serialize_model,
    serialize_grad
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


def wait(reqs, i):
    if reqs[i] is not None:
        for r in reqs[i]:
            r.wait()
    reqs[i] = None

class Buffer:
    def __init__(self, n, dtype):
        self.buffers = [
            init_tensor(n, init_func=torch.zeros, dtype=dtype),
            init_tensor(n, init_func=torch.zeros, dtype=dtype),
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


def bind_flatten(model, tensor):
    i = 0
    for p in params(model):
        n = p.data.numel()
        p.data = tensor[i : i + n].view(p.data.shape)
        i += n

class WeiPipe:
    def __init__(self, config, gradient_accumulation_steps=1, train_embedding=False, dtype=torch.float16, dl_iter=None, batch_size=8):
        # Setup world info
        self.backward_stream = torch.cuda.Stream()
        self.data_stream = torch.cuda.Stream()
        self.batch_size = batch_size
        self.enable_checkpointing = config.checkpointing
        self.dl_iter = dl_iter
        
        # self.get_data()

        
        self.train_embedding = train_embedding

        
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.dtype = dtype

        self.reqs=[None, None, None]
        self.config = copy.deepcopy(config)

        config.n_layers //= self.world_size

        self.model_32 = Model(config).cuda()            
        # backward model
        self.model_16 = Model(config).cuda().to(dtype)
    
        self.X = torch.randint(0, config.vocab_size,(batch_size, config.max_seq_len)).cuda()
        self.Y_ = torch.randint(0, config.vocab_size,(batch_size, config.max_seq_len)).cuda()


        num_decoders_params = num_params(self.model_16.decoders)

        self.buffers = {
            "weight0": Buffer(num_decoders_params, dtype=dtype),
            "weight1": Buffer(num_decoders_params, dtype=dtype),
            "grad": Buffer(num_decoders_params,  dtype=dtype),
        }

        self.flatten_weight()

        self.loss_fn = loss_fn
        self.activations = ActivationBuffer()
        self.grad = None
        # trainable_modules = nn.ModuleList ([self.model_32.decoders, self.model_32.norm])
        
        if train_embedding:
            trainable_modules = self.model_32
        else:
            trainable_modules = self.model_32.decoders

        self.optimizer = configure_optimizers(trainable_modules)
        self.optimizer.zero_grad()

        self.counter = {}
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.pg0 = torch.distributed.new_group(backend="nccl")
        self.pg1 = torch.distributed.new_group(backend="nccl")
        
        self.warmup_pg = [torch.distributed.new_group(ranks=[0,1], backend="nccl"), torch.distributed.new_group(ranks=[0,1,2],backend="nccl")]

        self.n_forward = 0
        self.n_backward = 0
        self.first = True
        
    def destroy(self):
        dist.destroy_process_group()

        
    def send_recv(self, ops):
        req = dist.batch_isend_irecv(ops)
        return req
    
    def flatten_weight(self):
        i = 0
        for p in params(self.model_32.decoders):
            n = p.data.numel()
            self.buffers["weight0"].recv[i : i + n] = p.data.view(-1).to(self.dtype)
            i += n
            
        if self.train_embedding:
            self.num_params_embedding = num_params(self.model_16.embedding)
            self.num_params_norm = num_params(self.model_16.norm)
            self.embedding_grad_buffer = init_tensor(self.num_params_embedding + self.num_params_norm, dtype=self.dtype)
            copy_weights(self.model_32.output, self.model_16.output)
            copy_weights(self.model_32.norm, self.model_16.norm)
            
    # ring exchange
    def flow_op(self, idx, reverse, process_group=None):
        prev_rank = (self.rank + self.world_size - 1) % self.world_size
        next_rank = (self.rank + 1) % self.world_size
        if reverse:
            prev_rank, next_rank = next_rank, prev_rank

        buffer = self.buffers[idx]
        
        # if self.first:
        #     self.first = False
        #     prev_rank = 1-self.rank
        #     next_rank = 1-self.rank
        #     process_group = self.warmup_pg[0]
        
        if self.rank % 2 == 0:
            send_op = dist.P2POp(dist.isend, buffer.send, next_rank, group=process_group)
            recv_op = dist.P2POp(dist.irecv, buffer.recv, prev_rank, group=process_group)
            ops = [send_op, recv_op]
        else:
            recv_op = dist.P2POp(dist.irecv, buffer.recv, prev_rank, group=process_group)
            send_op = dist.P2POp(dist.isend, buffer.send, next_rank, group=process_group)
            ops = [recv_op, send_op]
        

        return ops
    

    def weight_flow(self, idx, reverse):
        if self.world_size == 1:
            return
        
        
        weight_buffer = self.buffers[f"weight{idx}"]
        weight_buffer.send.copy_(weight_buffer.recv)
        
        if (idx == 0 and self.n_forward % self.world_size == 0):
            i = 0
            for p in params(self.model_32.decoders):
                n = p.data.numel()
                self.buffers["weight0"].recv[i : i + n] = p.data.view(-1).to(self.dtype)
                i += n
            return
        elif (idx == 1 and self.n_backward % self.world_size == 0):
            i = 0
            for p in params(self.model_32.decoders):
                n = p.data.numel()
                self.buffers["weight1"].recv[i : i + n] = p.data.view(-1).to(self.dtype)
                i += n
            return
        else:
            process_group = self.pg0 if idx == 0 else self.pg1
            # process_group=None
            weight_flow_op = self.flow_op(f"weight{idx}", reverse, process_group=process_group)
            
            req = self.send_recv(weight_flow_op)

            return req

    def grad_swap(self):
        """At the end, swap grad between rank i and rank n-i"""
        dst_rank = (self.world_size + 1 - self.rank) % self.world_size

        if self.rank % 2 == 0:
            send_op = dist.P2POp(dist.isend, self.buffers["grad"].send, dst_rank)
            recv_op = dist.P2POp(dist.irecv, self.buffers["grad"].recv, dst_rank)
            return  self.send_recv([send_op, recv_op])
        else:
            recv_op = dist.P2POp(dist.irecv, self.buffers["grad"].recv, dst_rank)
            send_op = dist.P2POp(dist.isend, self.buffers["grad"].send, dst_rank)
            return  self.send_recv([recv_op, send_op])
    
    def _forward(self, compute=False):
        n = self.gradient_accumulation_steps * self.world_size + self.world_size-1
        self.n_forward = (self.n_forward + 1) % n
        
        wait(self.reqs, 0)
        
        if self.n_forward > 0:
            self.reqs[0] = self.weight_flow(0, reverse=False)
            
        if compute:
            self.forward_step()

    def _backward(self, compute=False):
        # with torch.cuda.stream (self.backward_stream):
            n = self.gradient_accumulation_steps * self.world_size + self.world_size-1
            self.n_backward = (self.n_backward + 1) % n
            wait(self.reqs, 1)
            
            if self.n_backward > 0:
                self.reqs[1] = self.weight_flow(1, reverse=True)
                
            if compute:
                self.backward_step()

    def grad_flow(self, send=True):
        wait(self.reqs, 2)
        grad_buffer = self.buffers["grad"]
        grad_to_tensor(self.model_16.decoders, grad_buffer.send)
        if send:
            self.reqs[2] = self.send_recv(self.flow_op("grad", reverse=False))
        self.buffers["grad"].pingpong()
            
    def forward_step(self):
        x = self.activations.top()
        x.retain_grad()

        bind_flatten(self.model_16.decoders, self.buffers["weight1"].send)
        ctx = nullcontext() if self.enable_checkpointing else torch.no_grad()
        
        with ctx:
            x = self.model_16(x)
        
        self.activations.push(x)
        self.activations.push(x, detach=True)

    def backward_step(self):
        bind_flatten(self.model_16.decoders, self.buffers["weight1"].send)
        outputs = self.activations.pop()
        inputs = self.activations.pop()
        # recomputation
        if not self.enable_checkpointing:
            outputs = self.model_16(inputs)
        
        # else:
        # replace weight and then do checkpointing
        
        outputs.backward(self.grad)
        self.grad = inputs.grad
    
    def preprocess(self, x):
        x = self.model_16.embedding(x)
        x = self.model_16.dropout(x)
        return x

    def postprocess(self, x):
        x = self.model_16.norm(x)
        x = self.model_16.output(x)
        return x

    def calc_grad(self):
        outputs = self.activations.pop()
        loss = self.loss_fn(self.postprocess(outputs), self.Y)
        loss.backward()
        self.grad = outputs.grad
        
        self.Y = self.Y_
        return loss

    
    def get_data(self):
        # with torch.cuda.stream(self.data_stream):
            # self.X, self.Y_ = next(self.dl_iter)
            # print(self.X.shape)
            if self.X is None:
                self.X = torch.randint(0,32,(self.batch_size, self.config.seq_len)).cuda()
                self.Y = torch.randint(0,32,(self.batch_size, self.config.seq_len)).cuda()
        
    def forward_backward_step(self):
        self.i = 0
        self.first = True
        self.print_string = f"rank{self.rank}: "
        
        x = self.preprocess(self.X)
        self.Y = self.Y_
        self.get_data()
        
        self.activations.push(x, detach=True)
        
        # warmup-idle
        for _ in range(self.rank):
            self._forward(False)

        # warmup-forward
        for i in range(self.world_size - self.rank):
            self._forward(True)
            if i == self.world_size - self.rank - 1:
                # fork a backward weight
                self.buffers[f"weight{1}"].recv.copy_(self.buffers[f"weight{0}"].send)
        
        def stage (do_f, do_b):
            self._backward(do_b)
            self.grad_flow() 
            self._forward(do_f)
            
        warmup_stage = lambda : stage(True, False)
        acc_stage = lambda : stage(True, True)
        cooldown_stage = lambda : stage (False, True)
            
        for i in range(self.rank):
            warmup_stage()
            
        x1 = x
        for i in range(self.gradient_accumulation_steps - 1):
            self.activations.reverse()
            loss = self.calc_grad()
            # torch.cuda.current_stream().wait_stream(self.forward_stream)
            
            x1 = self.preprocess(self.X)
            self.get_data()
            
            self.activations.push(x1, detach=True)

            for _ in range(self.world_size):
                acc_stage()

            if i == self.gradient_accumulation_steps - 2 and  self.train_embedding:
                x.backward(self.grad)
        
        x = x1
        self.activations.reverse()

        loss = self.calc_grad()

        for i in range(self.world_size - 1 - self.rank):
            cooldown_stage()
            
        for i in range(self.world_size):
            if i <= self.rank:
                self._backward(True)
            else:
                self._backward(False)

            self.grad_flow(i != self.world_size-1)

        if self.train_embedding:
            x.backward(self.grad)

        self.buffers["grad"].pingpong()
        
        wait([self.grad_swap()], 0)
        assert self.reqs[0] is None
        assert self.reqs[1] is None
        assert self.reqs[2] is None
        self.activations.reverse()        
        self.update()
        

        return loss

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
            
    def update(self):
        # exchange params for embedding

        if self.train_embedding:
            self.embedding_grad_buffer.zero_()
            grad_to_tensor(self.model_16.embedding, self.embedding_grad_buffer[0:self.num_params_embedding])
            grad_to_tensor(self.model_16.norm, self.embedding_grad_buffer[self.num_params_embedding:])
            
            if self.world_size > 1:
                dist.all_reduce(self.embedding_grad_buffer)

            self.embedding_grad_buffer /= self.world_size * self.gradient_accumulation_steps
            
            tensor_to_grad(self.embedding_grad_buffer[0:self.num_params_embedding], self.model_32.embedding)
            tensor_to_grad(self.embedding_grad_buffer[self.num_params_embedding:], self.model_32.norm)
        
        self.buffers["grad"].recv /= self.world_size * self.gradient_accumulation_steps
        tensor_to_grad(
            self.buffers["grad"].recv,
            self.model_32.decoders,
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.buffers["grad"].send.zero_()

        # copy model32 to model16
        i = 0
        for p in params(self.model_32.decoders):
            n = p.data.numel()
            self.buffers["weight0"].recv[i : i + n] = p.data.view(-1).to(self.dtype)
            i += n
        
        if self.train_embedding:
            copy_weights(self.model_32.output, self.model_16.output)
            copy_weights(self.model_32.norm, self.model_16.norm)

            
