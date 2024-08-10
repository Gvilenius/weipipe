import os
import json
import argparse
import sys


# batch_sizes = [72, 144, 216, 288]
# acc_steps=[6, 12, 18, 24]

parser = argparse.ArgumentParser()
parser.add_argument("--ngpu", default=8, type=int)
parser.add_argument("--algo", default="zb", type=str, choices=["zb1", "zb2", "wei", "ds", "1f1b", "1f1bi"])

args = parser.parse_args()

ngpu_per_node = args.ngpu
nnode = 1

def set_env(k, v):
    os.environ[k] = str(v)


p = nnode * ngpu_per_node

def init(config):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"
    set_env("ALGO", args.algo)
    set_env("PIPELINE_SIZE", p)
    set_env("GPUS_PER_NODE", ngpu_per_node)
    set_env("LAYERS", config["n_layers"] * p)
    set_env("MICRO_BATCH_SIZE", config["batch_size"])
    set_env(
        "GLOBAL_BATCH_SIZE",
        config["gradient_accumulation_steps"] * config["batch_size"] * p,
    )
    set_env("HIDDEN_SIZE", config["dim"])
    set_env("ATTENTION_HEADS", config["n_heads"])
    set_env("SEQ_LEN", config["max_seq_len"])
    set_env("EXIT_INTERVAL", config["iters_num"])
    # set_env("INTERLEAVED_1F1B", 1)

with open("../weipipe/config.json", "r") as f:
    config = json.load(f)

init(config)


if args.algo == "zb1":
    set_env("ENABLE_ZERO_BUBBLE", 1)
    set_env("ZERO_BUBBLE_MEM_LIMIT", 1 * p)
    os.system(
        f"cd ../zero-bubble-pipeline-parallelism && bash examples/pretrain_zero_bubble.sh"
    )
elif args.algo == "zb2":
    set_env("ENABLE_ZERO_BUBBLE", 1)
    set_env("ZERO_BUBBLE_MEM_LIMIT", 2 * p)
    os.system(
        f"cd ../zero-bubble-pipeline-parallelism && bash examples/pretrain_zero_bubble.sh"
    )
elif args.algo == "1f1b":
    os.system(
        f"cd ../zero-bubble-pipeline-parallelism && bash examples/pretrain_zero_bubble.sh"
    )
elif args.algo == "1f1bi":
    set_env("INTERLEAVED_1F1B", 1)
    os.system(
        f"cd ../zero-bubble-pipeline-parallelism && bash examples/pretrain_zero_bubble.sh"
    )

elif args.algo == "ds":
    os.system(f"torchrun --nproc-per-node={ngpu_per_node} --nnodes={nnode} train-ds.py")
else:
    os.system(f"torchrun --nproc-per-node={ngpu_per_node} --nnodes={nnode} train-weipipe.py")
