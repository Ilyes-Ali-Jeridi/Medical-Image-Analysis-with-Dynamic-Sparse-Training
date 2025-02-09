#!/bin/bash

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: CUDA is not available on this system"
    exit 1
fi

# Set NCCL optimization variables
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=^docker0,lo

# Set Python optimizations
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# Configure thread optimization for NC48ads
export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48

# Set distributed training variables
export MASTER_ADDR="localhost"
export MASTER_PORT="29500"

# Launch distributed training
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/main.py \
    --train-csv /data/MIMICCXR/train_data.csv \
    --val-csv /data/MIMICCXR/val_data.csv \
    --epochs 10 \
    --cache-dir /tmp/mimic_cache