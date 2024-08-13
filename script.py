import os
import csv
import argparse
from utils import get_env, set_env
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,7'



parser = argparse.ArgumentParser()
parser.add_argument("--ngpu", default=8, type=int)
parser.add_argument("--algo", default="zb1", type=str, choices=["zb1", "zb2", "wei", "ds", "1f1b", "ddp"])
parser.add_argument("--rank", default=0, type=int)
parser.add_argument("--nnode", default=1, type=int)

parser.add_argument("--mode", default="single", type=str, choices=["single", "all", "scale"])

args = parser.parse_args()


ngpu_per_node = args.ngpu // args.nnode
master_addr = "localhost"
master_port = "10086"

# launch args
set_env ("MASTER_ADDR", master_addr)
set_env ("MASTER_PORT", master_port)
set_env ("RANK", args.rank)
set_env ("WORLD_SIZE", args.nnode)
set_env ("GPUS_PER_NODE", ngpu_per_node)
set_env ("PIPELINE_SIZE", args.ngpu)

# model args
set_env("HIDDEN_SIZE", 1024)
set_env("ATTENTION_HEADS", 32)
set_env("SEQ_LEN", 2048)
set_env("LAYERS", args.ngpu*2)

# training args
set_env("MICRO_BATCH_SIZE", 6)
set_env("ACC_STEP", 2)
set_env("CHECKPOINTING", 1)
set_env("TRAIN_EMBEDDING", 0)
set_env("EXIT_INTERVAL", 2)
set_env(
    "GLOBAL_BATCH_SIZE",
    get_env("MICRO_BATCH_SIZE") * get_env("ACC_STEP") * args.ngpu,
)

def run_single(algo, ngpu_per_node):
    set_env("ALGO", algo)     
    if os.environ["ALGO"] == "zb1":
        set_env("ENABLE_ZERO_BUBBLE", 1)
        set_env("ZERO_BUBBLE_MEM_LIMIT", 1 * args.ngpu)
        os.system(
            "cd ../zero-bubble-pipeline-parallelism && bash examples/pretrain_zero_bubble.sh"
        )
    elif os.environ["ALGO"] == "zb2":
        set_env("ENABLE_ZERO_BUBBLE", 1)
        set_env("ZERO_BUBBLE_MEM_LIMIT", 2 * args.ngpu)
        os.system(
            "cd ../zero-bubble-pipeline-parallelism && bash examples/pretrain_zero_bubble.sh"
        )
    elif os.environ["ALGO"] == "1f1b":
        os.system(
            "cd ../zero-bubble-pipeline-parallelism && bash examples/pretrain_zero_bubble.sh"
        )
    # elif os.environ["ALGO"] == "1f1bi":
    #     set_env("INTERLEAVED_1F1B", 1)
    #     os.system(
    #         "cd ../zero-bubble-pipeline-parallelism && bash examples/pretrain_zero_bubble.sh"
    #     )
    elif os.environ["ALGO"] == "ds":
        os.system(f"torchrun --nproc-per-node={ngpu_per_node} --master-addr={master_addr} --master-port={master_port} --nnodes={args.nnode} --node-rank={args.rank} train-ds.py")
    elif os.environ["ALGO"] == "wei":
        os.system(f"torchrun --nproc-per-node={ngpu_per_node} --master-addr={master_addr} --master-port={master_port} --nnodes={args.nnode} --node-rank={args.rank} train-weipipe.py")
    else:
        os.system(f"torchrun --nproc-per-node={ngpu_per_node} --master-addr={master_addr} --master-port={master_port} --nnodes={args.nnode} --node-rank={args.rank} train-ddp.py")
    
def run_scale():


    set_env("LAYERS", 8)
    for ngpu_per_node in [2, 4, 8]:
        set_env ("PIPELINE_SIZE", ngpu_per_node)
        set_env ("GPUS_PER_NODE", ngpu_per_node)
        set_env("MICRO_BATCH_SIZE", ngpu_per_node)
        run_single("wei", ngpu_per_node=ngpu_per_node)


def run_all(ngpu_per_node):
    algos = ["zb1", "zb2", "wei", "ds", "1f1b"]
    for algo in algos:
        run_single(algo, ngpu_per_node=ngpu_per_node)
        
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


    print_mem()
    print()
    print_time()

if __name__ == "__main__":
    if args.mode == "single":
        run_single(args.algo)
    elif args.mode == "all":
        run_all()
    elif args.mode == "scale":
        run_scale()
    