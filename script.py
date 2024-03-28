import os

os.system(
    'torchrun --nproc-per-node=3 --nnodes=2 --node-rank=0 --master-addr="10.233.99.178" --master-port=8887  train.py'
)
