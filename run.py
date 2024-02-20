from weipipe import WeiPipe

import torch
from torch import nn
import torch.distributed.rpc as rpc
import tempfile
import torch.distributed as dist
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


torch.manual_seed(123)


class MyData(Dataset):
    def __init__(self):
        n_samples = 10
        seq_len = 3
        input = torch.randint(0, 10, (n_samples, seq_len + 1))
        enc_input = input[:, :-1].long()
        targets = input[:, 1:].long()
        self.data = list(zip(enc_input, targets))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--rank", default=-1, type=int)
args = parser.parse_args()


tmpfile = tempfile.NamedTemporaryFile()
rpc.init_rpc(
    name="worker",
    rank=0,
    world_size=1,
    rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
        init_method="file://{}".format(tmpfile.name),
        _transports=["ibv", "uv"],
        _channels=["cuda_ipc", "cuda_basic"],
    ),
)


def init_process_group():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())


init_process_group()

model = WeiPipe()
ds = MyData()
sampler = DistributedSampler(ds)
dataloader = DataLoader(ds, batch_size=2, sampler=sampler)
for x, y in dataloader:
    model.forward_backward_step(x.cuda().half(), y.cuda().half())
    break
