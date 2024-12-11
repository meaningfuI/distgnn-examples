# distgnn-examples
distributed GNN full-graph training examples based on DGL.

## environment
- python 3.10
- CUDA 11.6
- PyTorch 1.13.1
- DGL 2.0.0

## usage

multigpu_full.py: train GNN using multiple GPU on single node. use `python multigpu_full` to run. `--n_partitions` set number of partitions and used GPUs.

cluster_full.py: train GNN using torchrun to run on multiple node with single/multiple GPU.
for example:`torchrun --nnodes=4 --nproc_per_node=2 --node_rank=0 --master_addr=$YOUR_ADDR --master_port=29400 cluster_full.py`
