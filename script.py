import os
import json

rank = 0
for batch_size in [72, 144, 216, 288]:
    for acc_steps in [6, 12, 18, 24]:
        with open("config.json", "r") as f:
            config = json.load(f)
        with open("config.json", "w") as f:
            config["batch_size"] = batch_size
            config["gradient_accumulation_steps"] = acc_steps
            json.dump(config, f, indent=4)

    os.system(
        'torchrun --nproc-per-node=3 --nnodes=2 --node-rank={rank} --master-addr="10.233.99.178" --master-port=8887  train.py'
    )
    os.system(
        'torchrun --nproc-per-node=3 --nnodes=2 --node-rank={rank} --master-addr="10.233.99.178" --master-port=8887  run.py'
    )
