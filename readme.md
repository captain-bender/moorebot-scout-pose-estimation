# PoseYOLO - Moorebot Robot Pose Detection with YOLO11

A custom pose detection project using YOLO11 for detecting and tracking Moorebot robot keypoints for robotic applications and automation.

## Overview

This project implements robot pose detection using the YOLO11 pose estimation model specifically trained for Moorebot robots. It can detect robot poses in images and track keypoints for various applications such as robotic automation, quality control, industrial monitoring, and robot interaction systems.

## Features

- Custom training on Moorebot robot datasets
- Support for high-resolution images (up to 1920x1280)
- GPU acceleration with CUDA support
- Easy-to-use inference pipeline
- Visualization of detected robot poses with keypoints
- Comprehensive performance metrics and evaluation tools

## Project Structure

```
poseYolo/
├── training.py               # Model training script
├── test-evaluation.py             # Test dataset evaluation
├── test-individual.py             # Individual image testing
├── test-inference.py              # Inference testing
├── pose-metrics-analysis.py       # Comprehensive pose metrics
├── inference-timing.py            # Performance benchmarking
├── visualise_annotations.py       # Dataset visualization
├── visualise_annotations_arrows.py # Advanced visualization
├── dataset/                      # Training datasets
│   ├── moorebot_vX/              # Moorebot v2 dataset
│   │   ├── data.yaml             # Dataset configuration
│   │   ├── train/                # Training images and labels
│   │   ├── valid/                # Validation images and labels
│   │   └── test/                 # Test images and labels
├── runs/                         # Training outputs and results
│   └── moorebot_vX/              # Moorebot training results
├── misc/                         # Miscellaneous files
└── readme.md                     # This file
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Virtual environment (recommended)

### Setup

1. **Clone or download the project:**
   ```bash
   cd /path/to/your/project
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install ultralytics opencv-python torch torchvision
   ```

## Usage

### Training Custom Model

Train the model on your Moorebot dataset:

```bash
python training.py
```

### Model Evaluation

Evaluate your trained model on test data:

```bash
python test-evaluation.py      # Comprehensive test metrics
python test-individual.py      # Individual image testing
python pose-metrics-analysis.py # Detailed pose analysis
```

### Dataset Visualization

Visualize your training annotations:

```bash
python visualise-annotations.py        # Basic visualization
python visualise-annotations-arrows.py # Advanced visualization with arrows
```

**data.yaml configuration:**
```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

kpt_shape: [3, 3]  # 3 keypoints, 3 dimensions (x,y,visibility)
flip_idx: [2, 1, 0]

nc: 1
names: ['robot']
```

## Dependencies

- `ultralytics` - YOLO implementation
- `opencv-python` - Image processing and display
- `torch` - PyTorch deep learning framework
- `torchvision` - Computer vision utilities



