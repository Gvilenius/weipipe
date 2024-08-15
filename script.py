import os
import subprocess

import csv
import argparse
from utils import get_env, set_env

parser = argparse.ArgumentParser()
parser.add_argument("--ngpu", default=8, type=int)
parser.add_argument("--algo", default="scale", type=str, choices=["all", "scale", "zb1", "zb2", "wei", "ds", "1f1b", "ddp", "show", "base"])
parser.add_argument("--rank", default=0, type=int)
parser.add_argument("--nnode", default=1, type=int)
parser.add_argument("--master_addr", default="localhost", type=str)

args = parser.parse_args()
master_addr = args.master_addr
master_port = "10086"

# launch args
set_env ("MASTER_ADDR", master_addr)
set_env ("MASTER_PORT", master_port)

def init_env(ngpu, nnode):
    set_env ("PIPELINE_SIZE", ngpu)
    
    ngpu_per_node = ngpu // nnode
    set_env ("GPUS_PER_NODE", ngpu_per_node)
    
    set_env(
        "GLOBAL_BATCH_SIZE",
        get_env("MICRO_BATCH_SIZE") * get_env("ACC_STEP") * ngpu,
    )
    set_env ("WORLD_SIZE", args.nnode)
    

set_env ("RANK", args.rank)

# model args
set_env("HIDDEN_SIZE", 1024)
set_env("ATTENTION_HEADS", 32)
set_env("SEQ_LEN", 16384)


# training args
set_env("MICRO_BATCH_SIZE", 1)
set_env("ACC_STEP", 1)
set_env("CHECKPOINTING", "1")
set_env("TRAIN_EMBEDDING", 0)
set_env("EXIT_INTERVAL", 2)

set_env("PROF", 0)

def run_base():
    set_env("LAYERS", args.ngpu)
    for seq in [4096, 8192, 16384]:
        for acc_step in [4, 8]:
            for micro_batch_size in [4,8]:
                set_env("MICRO_BATCH_SIZE", micro_batch_size)
                set_env("ACC_STEP", acc_step)
                set_env("SEQ_LEN", seq)
                run_all(args.ngpu//args.nnode, args.nnode)

        

def run_single(algo, ngpu_per_node, nnode):
    set_env("ALGO", algo)    
    ngpu = ngpu_per_node * nnode
    
    init_env(ngpu=ngpu, nnode=nnode)
     
    if os.environ["ALGO"] == "zb1":
        set_env("ENABLE_ZERO_BUBBLE", 1)
        set_env("ZERO_BUBBLE_MEM_LIMIT", ngpu)
        cmd = "cd ../zero-bubble-pipeline-parallelism && bash examples/pretrain_zero_bubble.sh"
        
    elif os.environ["ALGO"] == "zb2":
        set_env("ENABLE_ZERO_BUBBLE", 1)
        set_env("ZERO_BUBBLE_MEM_LIMIT", 2 * ngpu)
        cmd = "cd ../zero-bubble-pipeline-parallelism && bash examples/pretrain_zero_bubble.sh"
        
    elif os.environ["ALGO"] == "1f1b":
        os.environ["ENABLE_ZERO_BUBBLE"]=""
        cmd = "cd ../zero-bubble-pipeline-parallelism && bash examples/pretrain_zero_bubble.sh"

    # elif os.environ["ALGO"] == "1f1bi":
    #     set_env("INTERLEAVED_1F1B", 1)
    #     os.system(
    #         "cd ../zero-bubble-pipeline-parallelism && bash examples/pretrain_zero_bubble.sh"
    #     )
    elif os.environ["ALGO"] == "ds":
        cmd = f"torchrun --nproc-per-node={ngpu_per_node} --master-addr={master_addr} --master-port={master_port} --nnodes={nnode} --node-rank={args.rank} train-ds.py"
    elif os.environ["ALGO"] == "wei":
        cmd = f"torchrun --nproc-per-node={ngpu_per_node} --master-addr={master_addr} --master-port={master_port} --nnodes={nnode} --node-rank={args.rank} train-weipipe.py"
    else:
        cmd = f"torchrun --nproc-per-node={ngpu_per_node} --master-addr={master_addr} --master-port={master_port} --nnodes={nnode} --node-rank={args.rank} train-ddp.py"
    
    try:
        process = subprocess.Popen(cmd, shell=True)
        process.communicate()
    except KeyboardInterrupt:
        import signal
        os.killpg(os.getpid(process.pid), signal.SIGKILL)
        exit()



def run_scale():
    nnode = 1
    set_env("LAYERS", 2)
    set_env("HIDDEN_SIZE", 1024)
    set_env("ATTENTION_HEADS", 32)
    set_env("MICRO_BATCH_SIZE", 1)
    set_env("ACC_STEP", 2)
    
    set_env("SEQ_LEN", 1024)
    for ngpu_per_node in [2]:
        run_all(ngpu_per_node=ngpu_per_node, nnode=nnode)


def run_all(ngpu_per_node, nnode):
    algos = ["zb1", "zb2", "wei", "ds", "1f1b"]
    for algo in algos:
        run_single(algo, ngpu_per_node=ngpu_per_node, nnode=nnode)
    show_all()
    
    
def show_all():
    def print_histogram(data):
        data = dict(sorted (data.items(), key=lambda kv:(kv[1], kv[0])))
        
        max_v = max(data.values())
        for k, v in data.items():
            bar = '*' * int( v / max_v * 50)  # 假设最大宽度为20
            print(f"{k}: {bar}")

    
    dir = "/workspace/weipipe/result"
    files = os.listdir(dir)
    files = [ f for f in files if f.endswith("csv") if f != "all.csv"]

    rs = []
    header = None

    for file in files:
        with open(os.path.join(dir, file) , "r") as f:
            reader = csv.reader(f, delimiter=',', quotechar='"' )
            
            if header is None:
                header = next(reader)
                
            r = None
            for row in reader:
                r = row
            rs.append (r + [os.path.splitext(file)[0]])
            
            # f.writerow (file, r[-2], r[-1])

    fname = os.path.join(dir, "all.csv")

    exist = os.path.exists(fname)
    with open(fname, "a") as f:

        writer = csv.writer(f)
        if not exist:
            writer.writerow(header + ["algo"])
            
        writer.writerow(rs[0])
        for r in rs[1:]:
            writer.writerow (["" for x in r[:-3]] + r[-3:])    


    def print_mem():
        print("mem")
        graph = dict()
        for r in rs:
            graph[r[-1].ljust(7, " ")] = float(r[-2])
        print_histogram(graph)

    def print_time():
        print("time")
        graph = dict()
        for r in rs:
            graph[r[-1].ljust(7, " ")] = float(r[-3])
        print_histogram(graph)


    # print_mem()
    # print()
    # print_time()

if __name__ == "__main__":
    set_env("LAYERS", args.ngpu)
    
    if args.algo == "all":
        run_all(args.ngpu, nnode=args.nnode)
    elif args.algo == "scale":
        run_scale()
    elif args.algo == "show":
        show_all()
    elif args.algo == "base":
        run_base()
    else:
        run_single(args.algo, ngpu_per_node=args.ngpu, nnode=args.nnode)
    
