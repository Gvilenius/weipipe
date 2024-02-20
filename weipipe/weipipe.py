import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from pprint import pprint


def models_to_tensor(models):
    return torch.hstack([model_to_tensor(model) for model in models])


def model_to_tensor(model):
    return torch.hstack(
        [p.data.flatten().to(torch.float16) for p in model.parameters()]
    )


def tensor_to_model(tensor, model):
    i = 0
    for p in model.parameters():
        n = p.data.numel()
        raw = tensor.narrow(0, i, n)
        i += n
        p.data = raw.reshape(p.data.shape)


def tensor_to_models(tensor, models):
    i = 0
    for model in models:
        for p in model.parameters():
            n = p.data.numel()
            raw = tensor.narrow(0, i, n)
            i += n
            p.data = raw.reshape(p.data.shape)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3, bias=False)
        data = dist.get_rank() * torch.ones(self.linear.weight.shape)
        self.linear.weight.data = data

    def forward(self, x):
        return self.linear(x)


class WeiPipe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Setup world info
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.model_fp32 = Layer().cuda().to(dtype=torch.float32)
        # self.model = self.model_fp32.to(dtype=torch.float16)
        self.models = [
            self.model_fp32.to(torch.float16),
            Layer().cuda().to(dtype=torch.float16),
        ]
        self.current_model_index = 0

        self.activations = []
        self.gradients = []
        self.flattened_model = [None, None]

        self.state = -1 * self.rank
        self.loss_fn = F.cross_entropy
        self.optimizer = torch.optim.SGD(
            self.model_fp32.parameters(), lr=0.1, momentum=0.9
        )
        self.optimizer.zero_grad()

    def weight_swap(self):
        """At the begining, swap weight between rank i and rank n-i"""
        dst_rank = self.world_size - 1 - self.rank

        self.flattened_model[0] = model_to_tensor(self.models[0])
        self.flattened_model[1] = (
            torch.empty(len(self.flattened_model[0])).half().cuda()
        )

        send_op = dist.P2POp(dist.isend, self.flattened_model[0], dst_rank)
        recv_op = dist.P2POp(dist.irecv, self.flattened_model[1], dst_rank)
        reqs = dist.batch_isend_irecv([send_op, recv_op])
        for r in reqs:
            r.wait()
        tensor_to_model(self.flattened_model[1], self.models[1])
        self.print_model(True)

    def weight_flow(self):
        prev_rank = (self.rank + self.world_size - 1) % self.world_size
        next_rank = (self.rank + 1) % self.world_size
        self.flattened_model[0] = models_to_tensor(self.models)
        self.flattened_model[1] = (
            torch.empty(len(self.flattened_model[0])).half().cuda()
        )
        send_op = dist.P2POp(dist.isend, self.flattened_model[0], next_rank)
        recv_op = dist.P2POp(dist.irecv, self.flattened_model[1], prev_rank)
        reqs = dist.batch_isend_irecv([send_op, recv_op])
        for r in reqs:
            r.wait()
        tensor_to_models(self.flattened_model[1], self.models)

    def is_pipeline_last_stage(self):
        return self.rank == self.world_size - 1

    def is_pipeline_first_stage(self):
        return self.rank == 0

    def forward(self, x):
        return self.models[self.current_model_index](x)

    def forward_model(self):
        self.current_model_index = 0

    def backward_model(self):
        self.current_model_index = 1

    def print_model(self, all=False):
        model_str = str(*self.models[self.current_model_index].parameters())
        if self.current_model_index == 0:
            msg = f"rank{self.rank} forward using {model_str}"
        else:
            msg = f"rank{self.rank} backward using {model_str}"

        if all:
            model_str0 = str(*self.models[0].parameters())
            model_str1 = str(*self.models[1].parameters())
            msg = f"rank{self.rank} all model parameter is {model_str0} {model_str1}"

        print(msg)

    def forward_backward_step(self, inputs, targets=None):
        self.weight_swap()
        self.activations.append(inputs)
        for i in range(self.world_size * 3):
            i_offset = i - self.rank

            # calculate loss
            if i_offset == self.world_size:
                outputs = self.activations.pop()
                outputs.requires_grad = True
                self.loss_fn(outputs, targets).backward()
                grad = outputs.grad

            # forward
            if 0 <= i_offset < self.world_size:
                self.forward_model()
                with torch.no_grad():
                    x = self.activations[-1]
                    y = self.forward(x)
                self.activations.append(y)
            # backward
            elif self.world_size <= i_offset < self.world_size * 2:
                self.backward_model()
                inputs = self.activations.pop().detach()
                inputs.requires_grad = True

                outputs = self.forward(inputs)
                outputs.backward(grad)
                grad = inputs.grad
            self.weight_flow()
        # backward *world_size* times
        # recomputation

        return outputs

    def backward(self, x):
        x = self.activations.pop().detach()
        self.forward_step(x)
        y = self.activations.pop()
        loss = self.loss_fn(y)
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    pass
