# Medical Image Analysis with Dynamic Sparse Training

A state-of-the-art system for medical image analysis using Dynamic Sparse Training (DST) for efficient processing of radiological images and automated report generation.

## Project Overview

This project implements an advanced medical imaging system that combines:
- Dynamic Sparse Training for efficient model training
- Vision Transformers for X-ray image analysis
- Retrieval-Augmented Generation for report creation
- Multi-GPU distributed training optimized for Azure NC48ads A100 v4

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Dataset Setup](#dataset-setup)
5. [Training](#training)
6. [Monitoring](#monitoring)
7. [Evaluation](#evaluation)
8. [Deployment](#deployment)
9. [Troubleshooting](#troubleshooting)

## System Requirements

- Azure NC48ads A100 v4 instance (or similar with NVIDIA A100 GPUs)
- Recommended: 4x NVIDIA A100 GPUs (40GB each)
- Recommended: 440GB System RAM
- Ubuntu 20.04 or later
- CUDA 11.8 or later (compatible with PyTorch version)
- Python 3.10.x is recommended.

## Project Structure

```
.
├── requirements.txt    # Project dependencies
├── launch.sh          # Training launch script
└── src/
    ├── config.py      # Configuration management
    ├── dataset.py     # MIMIC-CXR data handling
    ├── models.py      # Core model implementations
    ├── optimizer.py   # A100 GPU optimizations
    ├── trainer.py     # Distributed training
    ├── evaluator.py   # Performance metrics
    ├── deployer.py    # Model deployment
    └── main.py        # Entry point
```

### Component Details

1.  **`src/config.py`**: Centralized configuration for paths, hardware, model hyperparameters, learning rates, and training settings.
2.  **`src/dataset.py`**: Handles the MIMIC-CXR dataset, including preprocessing, caching, and a `DEBUG_MODE` for quick testing using a subset of data (activated by setting the `DEBUG_MODE=True` environment variable).
3.  **`src/models.py`**:
    *   `DynamicSparseLayer`: Implements adaptive sparse training for dynamic pruning and regrowing of connections.
    *   `MedicalViT`: Vision Transformer adapted for X-ray image analysis.
    *   `MedicalRAG`: Retrieval-Augmented Generation model for report creation. Includes FAISS index management for semantic search, with options for index persistence (`faiss_index_path`, `report_db_path` in `Config`).
4.  **`src/optimizer.py`**:
    *   `A100Optimizer`: Contains static methods to configure PyTorch for optimal performance on NVIDIA A100 GPUs (and other modern GPUs), including TF32 enablement, cuDNN settings, and distributed training setup. Manages GPU memory fraction (`cuda_memory_fraction`) and PyTorch threading (`torch_num_threads`) via `Config`.
5.  **`src/trainer.py`**:
    *   `DistributedRadiologyTrainer`: Orchestrates distributed training with DDP, Automatic Mixed Precision (AMP), gradient accumulation, and checkpointing (periodic and best model based on validation loss, configurable via `checkpoint_save_every_n_epochs` and `save_best_checkpoint` in `Config`).
    *   **Critical Note**: This component has known issues regarding the complete and correct implementation of gradient accumulation and refined checkpointing logic (see [Troubleshooting](#troubleshooting)).
6.  **`src/evaluator.py`**: Calculates BLEU, CLIP, and CheXbert scores for model evaluation.
7.  **`src/deployer.py`**: Handles ONNX model export, inference optimization, and batch prediction.
8.  **`src/main.py`**: Main entry point, parses command-line arguments (which can override `Config` settings for paths like `--data-dir`, `--output-dir`, etc.), sets up logging, and coordinates the training, evaluation, and deployment phases.

## Installation

1. **System Setup**
   ```bash
   # Update system
   sudo apt-get update && sudo apt-get upgrade -y
   
   # Install system dependencies
   sudo apt-get install -y build-essential git wget curl nvidia-cuda-toolkit
   ```

2. **Python Environment**
   ```bash
   # Install Miniconda
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh -b
   echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc

   # Create environment
   conda create -n medical_env python=3.10 -y
   conda activate medical_env
   ```

3. **Install Dependencies**
   ```bash
   # Install PyTorch
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

   # Install other requirements
   pip install -r requirements.txt
   ```

## Dataset Setup

### MIMIC-CXR Dataset Access

1. **Prerequisites**
   - Complete PhysioNet credentialing process
   - Visit: https://physionet.org/content/mimic-cxr/2.0.0/
   - Obtain access credentials

2. **Directory Setup**
   ```bash
   mkdir -p /data/MIMICCXR/{images,reports}
   cd /data/MIMICCXR
   ```

3. **Download Dataset**
   ```bash
   # Download images (replace with your credentials)
   wget -r -N -c -np --user <username> --ask-password \
   https://physionet.org/files/mimic-cxr/2.0.0/files/

   # Download reports
   wget -r -N -c -np --user <username> --ask-password \
   https://physionet.org/files/mimic-cxr/2.0.0/reports/
   ```

4. **Organize Data**
   ```bash
   # Create organization script
   cat > organize_data.sh << 'EOF'
   #!/bin/bash
   
   # Move image files
   find physionet.org/files/mimic-cxr/2.0.0/files/ -name "*.dcm" | while read file; do
     rel_path=${file#physionet.org/files/mimic-cxr/2.0.0/files/}
     mkdir -p "images/$(dirname $rel_path)"
     mv "$file" "images/$rel_path"
   done

   # Move report files
   find physionet.org/files/mimic-cxr/2.0.0/reports/ -name "*.txt" | while read file; do
     rel_path=${file#physionet.org/files/mimic-cxr/2.0.0/reports/}
     mkdir -p "reports/$(dirname $rel_path)"
     mv "$file" "reports/$rel_path"
   done
   EOF

   chmod +x organize_data.sh
   ./organize_data.sh
   ```

5. **Generate CSV Files**
   ```bash
   python3 src/tools/create_dataset_csv.py \
     --image-dir /data/MIMICCXR/images \
     --report-dir /data/MIMICCXR/reports \
     --output-dir /data/MIMICCXR
   ```

## Training

1. **Environment Setup**
   ```bash
   # Configure CUDA devices (example for 8 GPUs)
   export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" 
   
   # Master address and port for distributed training (if not using torchrun's defaults)
   export MASTER_ADDR="localhost"
   export MASTER_PORT="29500"

   # Recommended NCCL environment variables for potentially better performance,
   # especially on multi-node setups or specific network configurations.
   # These should be set in your shell or launch.sh script.
   # export NCCL_DEBUG="INFO" # Or "WARN" for less verbosity
   # export NCCL_IB_DISABLE="0" # Set to 1 to disable InfiniBand if issues occur
   # export NCCL_IB_GID_INDEX="3" # May vary depending on InfiniBand setup
   # export NCCL_SOCKET_IFNAME="^docker0,lo" # Exclude docker and loopback interfaces
   ```

2. **Start Training**
   The `launch.sh` script is the recommended way to start distributed training.
   ```bash
   # Make launch script executable
   chmod +x launch.sh

   # Start distributed training using torchrun (adjust nproc_per_node as per your GPUs)
   # The launch.sh script should internally call src/main.py with appropriate arguments.
   # Example content for launch.sh:
   # #!/bin/bash
   # torchrun --nproc_per_node=4 --nnodes=1 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
   # src/main.py --train-csv train_split.csv --val-csv val_split.csv --epochs 10 --eval --deploy \
   # --output-dir ./outputs_$(date +%Y%m%d_%H%M%S) \
   # --log-file training_run.log 
   # # Add other CLI arguments as needed.

   ./launch.sh
   ```

3. **Training Options (Command-Line Overrides)**
   The `src/main.py` script accepts various command-line arguments that can override settings from `src/config.py`.
   ```bash
   # Basic training with specific CSVs (epochs default to value in Config)
   python src/main.py --train-csv train_split.csv --val-csv val_split.csv

   # Override epochs, output directory, and enable evaluation & deployment
   python src/main.py \
     --train-csv train_split.csv \
     --val-csv val_split.csv \
     --epochs 20 \
     --eval \
     --deploy \
     --data-dir /path/to/your/MIMICCXR \
     --output-dir /path/to/your/outputs \
     --checkpoint-dir /path/to/your/checkpoints \
     --cache-dir /path/to/your/cache \
     --log-file training_custom.log
   ```
   Refer to `src/main.py` (`parser.add_argument` calls) and `src/config.py` for all available options.

### Configuration Highlights
Key configuration settings managed in `src/config.py` (and overridable via CLI where applicable):
- **Hardware**: `num_gpus`, `batch_size`, `gradient_accumulation_steps`, `num_workers`, `cuda_memory_fraction`, `torch_num_threads`.
- **Model**: `image_size`, `patch_size`, `embed_dim`, `num_layers`, `sparse_rate`.
- **Optimizer**: `base_lr` (and derived `lr`), `weight_decay`.
- **Paths**: `data_dir`, `output_dir`, `checkpoint_dir`, `cache_dir`, `log_file`.
- **Checkpointing**: `checkpoint_save_every_n_epochs`, `save_best_checkpoint`.
- **Dataset**: `DEBUG_MODE` (environment variable `DEBUG_MODE=True` activates dataset debugging, using a smaller subset).
- **FAISS Index**: `faiss_index_path`, `report_db_path` (in `MedicalRAG` constructor, typically set via `Config` if used globally).

## Monitoring

1. **GPU Monitoring**
   ```bash
   # Real-time GPU stats
   watch -n 1 nvidia-smi

   # GPU utilization history
   nvidia-smi -l 1
   ```

2. **Training Progress**
   ```bash
   # View training logs (path configured in config.py or via --log-file)
   tail -f outputs/training.log # Or your custom log file path

   # Monitor with Weights & Biases (if enabled)
   # Ensure you have logged in:
   # wandb login
   # Then view your project dashboard on the WandB website.
   ```

3. **System Monitoring**
   ```bash
   # Process monitoring
   htop

   # Disk usage
   df -h

   # Memory usage
   free -h

   # Network usage
   iftop
   ```

## Evaluation

The system provides three evaluation metrics:
- BLEU score for report generation quality
- CLIP score for image-text alignment
- CheXbert score for medical accuracy

```bash
# Run evaluation
python src/main.py --eval \
  --checkpoint-dir checkpoints \
  --val-csv val.csv
```

## Deployment

1. **Export Model**
   ```bash
   python src/main.py --deploy \
     --checkpoint-dir checkpoints \
     --output-dir deployment
   ```

2. **Inference Optimization**
   - Automatic TorchScript compilation
   - ONNX export for deployment
   - Batch prediction support

## Troubleshooting

**CRITICAL KNOWN ISSUES (Require Manual Fixes):**
- **Gradient Accumulation**: The gradient accumulation logic in `src/trainer.py` (within the `train` method and `train_step`) was not fully corrected in the last automated refactoring cycle. It requires careful review and modification to ensure gradients are correctly scaled, accumulated over the specified number of steps, and that optimizer steps/mask updates occur only after accumulation.
- **Checkpointing Logic**: While configuration options for periodic and best-model checkpointing were added to `src/config.py`, the corresponding logic in `src/trainer.py` (specifically in the `train` and `save_checkpoint` methods) was not fully implemented or verified. This needs to be completed to ensure reliable checkpoint saving.

1. **Memory Issues (CUDA OOM)**
   - Reduce `batch_size` in `src/config.py`.
   - Increase `gradient_accumulation_steps` in `src/config.py` (this effectively reduces memory per step at the cost of less frequent updates).
   - Reduce `cuda_memory_fraction` in `src/config.py` if using `set_per_process_memory_fraction` (primarily for A100s).
   - Monitor GPU memory usage with `nvidia-smi`.

2. **Training Issues**
   - Check the log file (default: `outputs/training.log`) for detailed error messages.
   - Verify dataset paths in `src/config.py` (or overridden via CLI) are correct and accessible.
   - Ensure image and report files are correctly organized as per [Dataset Setup](#dataset-setup).
   - Confirm GPU availability and that `CUDA_VISIBLE_DEVICES` is set correctly if used.

3. **Performance Issues**
   - Adjust `num_workers` in `src/config.py` for optimal data loading.
   - Check GPU utilization with `nvidia-smi`. If low, data loading or CPU bottlenecks might be an issue.
   - Verify CUDA and cuDNN versions are compatible with your PyTorch build.
   - For multi-GPU, ensure NCCL environment variables are appropriately set for your system (see [Training - Environment Setup](#training)).

4. **NLTK Data Issues**:
   - If you see errors related to 'punkt' tokenizer (e.g., in `MedicalEvaluator`), ensure NLTK data was downloaded successfully at startup. The script attempts to download it and exit if critical, but network issues or permissions might interfere. Manually run `python -m nltk.downloader punkt` in your environment.

## Citation

```bibtex
@article{Ilyesalijeridi2025,
  title={Dynamic Sparse Training for Multi-Modal Radiology},
  author={Ilyes ali jeridi},
  year={2025}
}
```
