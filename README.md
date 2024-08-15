# Weigit Pipeline Parallelism

## Setup

```bash
docker pull nvcr.io/nvidia/pytorch:xx.xx-py3
mkdir workspace && cd workspace
git clone https://github.com/Gvilenius/zero-bubble-pipeline-parallelism.git
git clone https://github.com/Gvilenius/weipipe.git
git clone https://github.com/Gvilenius/DeepSpeed.git

docker run --gpus all --network host --ipc=host --shm-size=32G -it --rm -v /path/to/workspace:/workspace -v /path/to/dataset:/tmp/zb_sample_dataset -v nvcr.io/nvidia/pytorch:xx.xx-py3

cd weipipe && pip install -r requirements
cd ../Deepspeed && sh install.sh

```
