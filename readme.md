# moorebotPose - Moorebot Robot Pose Detection using YOLO11

A custom pose detection project using YOLO11 for detecting and tracking Moorebot scout robot keypoints for robotic applications and automation.

![](./misc/DSC04919.JPG)

## Overview

This project implements robot pose detection using the YOLO11 pose estimation model specifically trained for Moorebot robots. It can detect robot poses in images and track keypoints for various applications such as robotic automation, quality control, industrial monitoring, and robot interaction systems.

## Features

- Custom training on Moorebot robot datasets
- Support for high-resolution images (up to 1920x1280)
- GPU acceleration with CUDA support
- Easy-to-use inference pipeline
- Visualization of detected robot poses with keypoints
- Comprehensive performance metrics and evaluation tools
- Robot orientation calculation from keypoint triangles

## Quick Start

### Installation

**Prerequisites:** Python 3.8+, CUDA-capable GPU (recommended)

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate     # On Windows

# Install dependencies
pip install ultralytics opencv-python torch torchvision
```

### Training

Train the model on your Moorebot dataset:

```bash
python training.py
```

### Quick Inference

Run pose detection on new photos:

```bash
# Single image with orientation calculation
python run_inference.py --source misc/DSC04919.JPG --calc-orientation

# Batch process with JSON export
python run_inference.py --source /path/to/photos --save-json
```

### Model Evaluation

Evaluate model performance:

```bash
python test-evaluation.py      # Comprehensive metrics
python test-individual.py      # Individual testing
```

## Documentation

- **[Model Evaluation & Metrics](METRICS.md)** - Comprehensive guide to evaluation metrics, testing procedures, and performance analysis
- **[Ad-hoc Inference Guide](INFERENCE.md)** - Complete reference for running inference on arbitrary images with `run_inference.py`

## Dataset Visualization

Visualize training annotations:

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

## Datasets

The datasets can be found in Roboflow Universe:
- [version 5](https://universe.roboflow.com/moorebot-scout/moorebot-pose-g9lqr/dataset/7): New images added and the same pre-processing and augmentation applied as in version 4
- [version 4](https://universe.roboflow.com/moorebot-scout/moorebot-pose-g9lqr/dataset/6): Data set curation performed to add preprocessing steps and augmentations
- [version 3](https://universe.roboflow.com/moorebot-scout/moorebot-pose-g9lqr/dataset/5): More images added, keypoints skeleton remained the same
- [version 2](https://universe.roboflow.com/moorebot-scout/moorebot-pose-g9lqr/dataset/2): Initial small dataset, skeleton definition updated, results more promising
- [version 1](https://universe.roboflow.com/moorebot-scout/moorebot-pose-g9lqr/dataset/1): Initial small dataset, keypoints skeleton very strict defined, results were not so encouraging

## Dependencies

- `ultralytics` - YOLO implementation
- `opencv-python` - Image processing and display
- `torch` - PyTorch deep learning framework
- `torchvision` - Computer vision utilities


