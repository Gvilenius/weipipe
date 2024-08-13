import os
import json
import argparse
import sys




nnode = int(os.environ["WORLD_SIZE"])
node_rank = os.environ["RANK"]
master_addr = os.environ["MASTER_ADDR"]
master_port = os.environ["MASTER_PORT"]
ngpu_per_node = os.environ["GPUS_PER_NODE"]



def set_env(k, v):
    os.environ[k] = str(v)


p = nnode * ngpu_per_node

def init(config):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"
    set_env("CHECKPOINTING", config["checkpointing"])
    set_env("PIPELINE_SIZE", p)
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


