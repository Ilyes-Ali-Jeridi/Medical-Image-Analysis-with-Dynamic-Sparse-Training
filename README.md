# Medical Image Analysis with Dynamic Sparse Training

This project provides a state-of-the-art system for medical image analysis, leveraging dynamic sparse training, Vision Transformers (ViT), and Retrieval-Augmented Generation (RAG) to process radiological images and generate automated reports. The system is optimized for multi-GPU distributed training on high-performance infrastructure like the Azure NC48ads A100 v4, ensuring both efficiency and accuracy.

## Key Features

- **Dynamic Sparse Training (DST):** Reduces computational overhead by dynamically pruning model weights during training, leading to faster and more efficient model execution.
- **Vision Transformer (ViT):** Employs a state-of-the-art transformer-based architecture for high-accuracy medical image analysis.
- **Retrieval-Augmented Generation (RAG):** Enhances report generation by retrieving relevant information from a knowledge base, ensuring contextually rich and accurate reports.
- **Multi-GPU Optimization:** Built for distributed training on NVIDIA A100 GPUs, with optimizations for maximum performance.
- **Comprehensive Evaluation:** Includes a suite of metrics—BLEU, CLIP score, and CheXbert—to assess report quality, image-text alignment, and medical accuracy.

## Table of Contents

1. [Architecture](#architecture)
2. [Code Documentation](#code-documentation)
3. [Issues Fixed](#issues-fixed)
4. [System Requirements](#system-requirements)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Evaluation](#evaluation)
8. [Deployment](#deployment)
9. [Troubleshooting](#troubleshooting)
10. [Citation](#citation)

## Architecture

The system is designed with a modular architecture that separates concerns and promotes maintainability. The key components are:

- **Configuration (`config.py`):** Centralizes all hyperparameters, model settings, and paths, making it easy to configure and tune the system.
- **Data Loading (`dataset.py`):** Manages the MIMIC-CXR dataset, with support for caching to accelerate data loading during training.
- **Models (`models.py`):** Implements the core neural network architectures, including the `MedicalViT` for image analysis and the `MedicalRAG` for report generation.
- **Training (`trainer.py`):** Orchestrates the distributed training process, handling the training loop, validation, and checkpointing.
- **Optimization (`optimizer.py`):** Provides specialized optimizations for the NVIDIA A100 GPU, ensuring efficient training.
- **Evaluation (`evaluator.py`):** Calculates a suite of metrics to assess the performance of the generated reports.
- **Deployment (`deployer.py`):** Contains utilities for exporting the model to ONNX format and preparing it for inference.
- **Main (`main.py`):** The entry point for the application, tying all components together.

## Code Documentation

The codebase is comprehensively documented with docstrings in all major classes and functions. Below is a high-level overview of the modules in the `src` directory:

- **`config.py`:** Defines the `Config` class, which stores all hyperparameters and settings.
- **`dataset.py`:** Implements the `MIMICCXRDataset` class for loading and caching data.
- **`models.py`:** Contains the `MedicalViT` and `MedicalRAG` models, along with the `DynamicSparseLayer`.
- **`trainer.py`:** Implements the `DistributedRadiologyTrainer` class for managing the training process.
- **`optimizer.py`:** Provides the `A100Optimizer` class with static methods for GPU optimization.
- **`evaluator.py`:** Implements the `MedicalEvaluator` class for calculating performance metrics.
- **`deployer.py`:** Contains the `RadiologyDeployer` class for model deployment.
- **`main.py`:** The main script that parses arguments and runs the training, evaluation, or deployment pipeline.

## Issues Fixed

This version of the codebase includes several fixes and enhancements to improve stability, flexibility, and usability:

- **`NameError` in `config.py`:** Fixed by importing the `os` module.
- **Hardcoded Paths:** The data directory in `config.py` is now configurable via an environment variable.
- **Model Configurability:** Hardcoded model names and hyperparameters in `models.py` have been moved to the `Config` class.
- **Trainer Inconsistencies:** Resolved a learning rate mismatch and standardized model access in `trainer.py`.
- **Critical Errors in `dataset.py`:** Fixed conflicting class definitions, missing imports, and unsafe cache cleanup.
- **Syntax Error in `deployer.py`:** Corrected a nested class definition and streamlined model loading.
- **Comprehensive Documentation:** Added detailed docstrings to all classes and functions in the `src` directory.

## System Requirements

- **Hardware:**
  - Azure NC48ads A100 v4 instance (or equivalent)
  - 4x NVIDIA A100 GPUs (40GB each)
  - 440GB System RAM
- **Software:**
  - Ubuntu 20.04 or later
  - CUDA 11.8+
  - Python 3.10+
  - Miniconda

## Installation

1.  **System Setup:**
    ```bash
    sudo apt-get update && sudo apt-get upgrade -y
    sudo apt-get install -y build-essential git wget curl nvidia-cuda-toolkit
    ```

2.  **Python Environment:**
    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc

    conda create -n medical_env python=3.10 -y
    conda activate medical_env
    ```

3.  **Install Dependencies:**
    ```bash
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    ```

## Usage

The main entry point for the application is `src/main.py`, which is controlled via command-line arguments.

### Training

To start the training process, run the following command, replacing the CSV paths with your own:

```bash
python -m src.main --train-csv /path/to/train.csv --val-csv /path/to/val.csv --epochs 20
```

### Evaluation

To run evaluation after training, add the `--eval` flag:

```bash
python -m src.main --train-csv /path/to/train.csv --val-csv /path/to/val.csv --epochs 20 --eval
```

### Deployment

To export the model for deployment after training, add the `--deploy` flag:

```bash
python -m src.main --train-csv /path/to/train.csv --val-csv /path/to/val.csv --deploy
```

## Evaluation

The system uses three key metrics to evaluate the quality of the generated reports:

- **BLEU:** Measures the similarity between the generated and reference reports.
- **CLIP Score:** Assesses the alignment between the input image and the generated text.
- **CheXbert:** Evaluates the medical accuracy of the report by checking for the presence of key clinical findings.

## Deployment

The deployment process exports the vision encoder part of the model to the ONNX format, which is optimized for high-performance inference. To run deployment, use the `--deploy` flag as shown in the **Usage** section.

## Troubleshooting

- **Memory Issues:** If you encounter out-of-memory errors, try reducing the `batch_size` in `src/config.py`.
- **Training Failures:** Check `medical_training.log` for detailed error messages.
- **Performance Bottlenecks:** Ensure that the GPU drivers and CUDA version are up to date and that the system is configured as recommended.

## Citation

If you use this work in your research, please cite the following:

```bibtex
@article{Ilyesalijeridi2025,
  title={Dynamic Sparse Training for Multi-Modal Radiology},
  author={Ilyes ali jeridi},
  year={2025}
}
```