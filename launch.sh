#!/bin/bash

# Validate CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: CUDA devices not found"
    exit 1
fi

# Configure hardware optimizations
export NCCL_DEBUG=WARN
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONFAULTHANDLER=1

# Automatic batch size configuration
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
RECOMMENDED_BS=$(( 64 * NUM_GPUS ))

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --standalone \
    src/main.py \
    --train-csv /data/MIMICCXR/train_data.csv \
    --val-csv /data/MIMICCXR/val_data.csv \
    --epochs 10 \
    --batch-size $RECOMMENDED_BS \
    --gradient-accumulation-steps 2 \
    --mixed-precision bf16 \
    --cache-dir /tmp/mimic_cache
