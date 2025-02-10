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

- Azure NC48ads A100 v4 instance
- 4x NVIDIA A100 GPUs (40GB each)
- 440GB System RAM
- Ubuntu 20.04 or later
- CUDA 11.8+
- Python 3.10+

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

1. **models.py**
   - `DynamicSparseLayer`: Implements adaptive sparse training
   - `MedicalViT`: Vision Transformer for X-ray analysis
   - `MedicalRAG`: Report generation with retrieval augmentation

2. **trainer.py**
   - Distributed training implementation
   - Mixed precision training
   - Gradient scaling and accumulation
   - Checkpoint management

3. **evaluator.py**
   - BLEU score for text similarity
   - CLIP score for image-text alignment
   - CheXbert for medical accuracy

4. **deployer.py**
   - ONNX model export
   - Inference optimization
   - Batch prediction support

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
   # Configure CUDA devices
   export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
   export MASTER_ADDR="localhost"
   export MASTER_PORT="29500"
   ```

2. **Start Training**
   ```bash
   # Make launch script executable
   chmod +x launch.sh

   # Start distributed training
   ./launch.sh
   ```

3. **Training Options**
   ```bash
   # Basic training
   python src/main.py --train-csv train.csv --val-csv val.csv

   # Training with evaluation
   python src/main.py --train-csv train.csv --val-csv val.csv --eval

   # Training and deployment
   python src/main.py --train-csv train.csv --val-csv val.csv --eval --deploy
   ```

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
   # View training logs
   tail -f medical_training.log

   # Monitor with Weights & Biases
   wandb login
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

1. **Memory Issues**
   - Reduce batch size in `config.py`
   - Adjust gradient accumulation steps
   - Monitor GPU memory usage

2. **Training Issues**
   - Check `medical_training.log`
   - Verify dataset paths
   - Ensure GPU availability

3. **Performance Issues**
   - Adjust number of workers
   - Check GPU utilization
   - Verify CUDA version

## Citation

```bibtex
@article{Ilyesalijeridi2025,
  title={Dynamic Sparse Training for Multi-Modal Radiology},
  author={Ilyes ali jeridi},
  year={2025}
}
```
