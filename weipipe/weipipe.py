import os
import torch
import torch.nn as nn
import torch.distributed as dist
from pprint import pprint


def module_to_tensor(module):
    return torch.hstack ([p.data.flatten() for p in module.parameters()])

def tensor_to_module (tensor, module):
    i = 0
    for p in module.parameters():
        dim = p.data.dim()
        n = p.data.numel()
        raw = tensor.narrow (0, i, n)
        i += n

        # [1.0] -> [[1.0]]
        for _ in range (dim-1):
            raw = raw.unsqueeze(0)
        p.data = raw.reshape(p.data.shape)

def print_parameters(module):
    print ("rank:", dist.get_rank())
    for p in module.parameters():
        pprint(p)
    print()

class WeiPipe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Setup world info
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.model = nn.Sequential (
            nn.Linear(3,128),
            nn.Linear(128,128),
            nn.Linear(128,3)
        ).cuda()
        
        self.activations = []
        self.flattened_model = [module_to_tensor (self.model), None]
        self.flattened_model[1] = torch.empty (len(self.flattened_model[0])).cuda()
        self.state = - self.rank

    def forward (self, x):
        self.activations.append (x)
        for _ in range (self.world_size * 2 - 1):
            with torch.no_grad():
                self.forward_step (self.activations[-1])
        
        loss_fn = None
        loss  = loss_fn(self.activations.pop())

        self.backward(loss)
        print(self.activations)
        
    def backward(self, x):
        pass

    def forward_step(self, x):
        if self.state >= 0 and self.state < self.world_size:
            y = self.model(x)
            self.activations.append(y)
            
        self.state += 1
        
        self.flattened_model[0] = module_to_tensor(self.model)
        send_op = dist.P2POp(
            dist.isend, self.flattened_model[0], (self.rank + 1) % self.world_size
        )
        recv_op = dist.P2POp(
            dist.irecv,
            self.flattened_model[1],
            (self.rank + self.world_size - 1) % self.world_size,
        )
        reqs = dist.batch_isend_irecv([send_op, recv_op])
        for req in reqs:
            req.wait()
        tensor_to_module (self.flattened_model[1], self.model)

    def backward_step(self, dx):
        pass

if __name__ == "__main__":
    mlp = nn.Sequential (
        nn.Linear(2,3),
        nn.Linear(3,2)
    )
    x = torch.tensor([[1.0, 2.0]])
    t = module_to_tensor(mlp)

    print(mlp(x))
    tensor_to_module (t, mlp)
    print(mlp(x))
