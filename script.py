import os
import csv

def print_histogram(data):
    data = dict(sorted (data.items(), key=lambda kv:(kv[1], kv[0])))
    
    max_v = max(data.values())
    for k, v in data.items():
        bar = '*' * int( v / max_v * 50)  # 假设最大宽度为20
        print(f"{k}: {bar}")
        
os.system("python run.py --ngpu=8 --algo=1f1b")
os.system("python run.py --ngpu=8 --algo=1f1bi")
os.system("python run.py --ngpu=8 --algo=zb1")
os.system("python run.py --ngpu=8 --algo=zb2")
os.system("python run.py --ngpu=8 --algo=ds")
os.system("python run.py --ngpu=8 --algo=wei")



        
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