import torch
from torch import nn
import torch.distributed.rpc as rpc
import tempfile

tmpfile = tempfile.NamedTemporaryFile()
rpc.init_rpc(
    name="worker",
    rank=0,
    world_size=1,
    rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
        init_method="file://{}".format(tmpfile.name),
        # Specifying _transports and _channels is a workaround and we no longer
        # will have to specify _transports and _channels for PyTorch
        # versions >= 1.8.1
        _transports=["ibv", "uv"],
        _channels=["cuda_ipc", "cuda_basic"],
    )
)

from weipipe import WeiPipe

module_list = [nn.Linear (10,10).to(i) for i in range (3)]
model = WeiPipe(nn.Sequential(*module_list))

x = torch.Tensor([1.0]*10).cuda()
y_ref = model(x)

print(x, y_ref.local_value())