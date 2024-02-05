from weipipe import WeiPipe

import torch
from torch import nn
import torch.distributed.rpc as rpc
import tempfile
import torch.distributed as dist
import argparse

# tmpfile = tempfile.NamedTemporaryFile()
# rpc.init_rpc(
#     name="worker",
#     rank=0,
#     world_size=1,
#     rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
#         init_method="file://{}".format(tmpfile.name),
#         _transports=["ibv", "uv"],
#         _channels=["cuda_ipc", "cuda_basic"],
#     )
# )


parser = argparse.ArgumentParser()
parser.add_argument ("--local_rank", default=-1, type=int)
parser.add_argument ("--rank", default=-1, type=int)
args = parser.parse_args()
dist.init_process_group (
    backend="nccl"
)

torch.cuda.set_device (dist.get_rank())




model = WeiPipe()

with torch.no_grad():
    model.forward(torch.tensor([1.0, 2.0, 3.0]).cuda())

# x = torch.Tensor([1.0]*10).cuda()
# y_ref = model(x)

# print(x, y_ref.local_value())