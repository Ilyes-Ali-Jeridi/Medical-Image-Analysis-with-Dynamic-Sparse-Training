#!/bin/bash

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: CUDA is not available on this system"
    exit 1
fi

# Get number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

# Set environment variables
export MASTER_ADDR="localhost"
export MASTER_PORT="29500"

# Launch distributed training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/main.py \
    --train-csv /data/MIMICCXR/train_data.csv \
    --val-csv /data/MIMICCXR/val_data.csv \
    --epochs 10