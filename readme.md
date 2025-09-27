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

### Quick Inference on New Photos (Ad-hoc images)

Run pose inference on arbitrary photos that are not inside the dataset structure using the new `run_inference.py` utility.

Basic examples:
```bash
# Single image
python run_inference.py --source misc/DSC04919.JPG --model runs/moorebot_v5/train-v1/yolo11n-moorebot_v5-pose-v1/weights/best.pt

# All JPG images inside a folder
python run_inference.py --source /path/to/new_photos --model runs/moorebot_v5/train-v1/yolo11n-moorebot_v5-pose-v1/weights/best.pt

# Glob pattern (quote globs so the shell doesn't expand unexpectedly)
python run_inference.py --source 'new_photos/**/*.png' --model runs/moorebot_v5/train-v1/yolo11n-moorebot_v5-pose-v1/weights/best.pt

# Save JSON with structured detections (bboxes + keypoints)
python run_inference.py --source misc --save-json --out inference_outputs/moorebot_samples \
      --model runs/moorebot_v5/train-v1/yolo11n-moorebot_v5-pose-v1/weights/best.pt

# Grayscale (in-memory conversion) + show annotated window
python run_inference.py --source misc --grayscale --show \
      --model runs/moorebot_v5/train-v1/yolo11n-moorebot_v5-pose-v1/weights/best.pt

# Use each image's native resolution (no uniform square resize)
python run_inference.py --source misc --native \
   --model runs/moorebot_v5/train-v1/yolo11n-moorebot_v5-pose-v1/weights/best.pt
```

Key options:
- `--source`: File, directory, or glob pattern.
- `--out`: Output directory (default auto timestamp under `inference_outputs/`).
- `--save-json`: Exports a structured JSON (`results.json`) with bbox + keypoints per detection.
- `--grayscale`: Converts each image to grayscale before inference (useful for robustness checks).
- `--imgsz`: Override inference size (default 640). Increase (e.g. 1280) for potentially higher accuracy at cost of speed.
- `--native`: Preserve each image's original (H,W) resolution (model still internally pads to stride).
- `--conf` / `--iou`: Adjust thresholds for filtering.
- `--max`: Limit number of images processed (0 = all).
- `--show`: Open a window displaying each annotated image (ESC to abort early).

JSON schema example (per run):
```json
[
   {
      "image": "sample.jpg",
      "width": 1920,
      "height": 1080,
      "detections": [
         {
            "bbox": [x1, y1, x2, y2],
            "score": 0.91,
            "class": 0,
            "keypoints": [
               {"x": 123.4, "y": 456.7, "conf": 0.95},
               {"x": 140.2, "y": 460.1, "conf": 0.93},
               {"x": 160.8, "y": 470.9, "conf": 0.90}
            ]
         }
      ]
   }
]
```

Outputs are written as annotated images plus optional JSON to the chosen output directory.

### Metrics

### 1) Pose Estimation Metrics
Evaluates the accuracy of the predicted robot poses against ground truth (GT) annotations:
- **mAP50**: Mean Average Precision at IoU=0.50.
- **mAP75**: Mean Average Precision at IoU=0.75.
- **mAP50-95**: Mean Average Precision at IoU=0.50 to 0.95 (incremental 0.05).

Notes:
- Ensure your test data is correctly formatted and includes GT annotations.

### 2) PCK (Percentage of Correct Keypoints)
Measures the percentage of correctly predicted keypoints:
- Overall and per keypoint at thresholds 0.1 and 0.2.

Notes:
- Requires correct GT keypoint annotations in the test set.

### 3) OKS (Object Keypoint Similarity)
Evaluates the similarity between predicted and GT keypoints, normalized by object area:
- OKS AP@[.50:.95]: mean AP over thresholds 0.50, 0.55, …, 0.95.
- OKS AP@0.50: lenient match criterion.
- OKS AP@0.75: stricter localization.

Notes:
- OKS thresholds can be adjusted in the script for different evaluation criteria.

### 4) Inference Time
Measures the time taken for the model to predict poses on the test set.

Notes:
- Ensure a consistent environment for timing (e.g., same hardware, no other heavy processes running).

### 5) Keypoint Confidence (new)
Aggregates confidences of predicted keypoints across the test set (no GT matching):
- Overall: count, mean, median, std, and share above thresholds (default: 0.3, 0.5, 0.7).
- Per-detection: average keypoint confidence per detected instance (mean/median).
- Per-keypoint: mean, count, and share above thresholds (e.g., >= 0.5) for each keypoint index (uses `KEYPOINT_NAMES`).

Notes:
- This summarizes predicted confidences only.

### Results Location

Outputs from testing are saved under a structured `tests/`:

- Pattern: `tests/<dataset>/<split>/<model>/<tag|timestamp>/`
- Example: `tests/moorebot_v4/test/yolo11n-moorebot_v5-pose-v1/smoke/DSC04824...jpg`

Control the final folder with `--tag` (falls back to a timestamp if omitted):

```bash
python test-individual.py --count 2 --tag smoke
```

### Grayscale Inference

You can run individual image tests in grayscale to evaluate robustness to lighting/texture:

```bash
python test-individual.py --grayscale
```

- By default, temporary grayscale copies are created under a split-specific temp folder (e.g., `dataset/<name>/<split>_gray_tmp`) and automatically deleted at the end of the run.
- To keep the grayscale temp images for inspection, add `--keep-temp`.

### Cleanup

If any grayscale temp folders remain (e.g., after an interrupted run), use the provided cleanup utility:

```bash
# Dry run: show what would be removed
python3 cleanup_gray_tmp.py --dry-run

# Actually remove found *_gray_tmp directories (non-interactive)
python3 cleanup_gray_tmp.py --yes

# Optional: limit scan to a specific root
python3 cleanup_gray_tmp.py --root dataset --yes
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

## Datasets
The datasets can be found in roboflow universe:
- [version 5](https://universe.roboflow.com/moorebot-scout/moorebot-pose-g9lqr/dataset/5): New images added and the same pre-processing and augmentation applied as in version 4.
- [version 4](https://universe.roboflow.com/moorebot-scout/moorebot-pose-g9lqr/dataset/4): Data set curation performed to add preprocessing steps and augmentations.
- [version 3](https://universe.roboflow.com/moorebot-scout/moorebot-pose-g9lqr/dataset/3): More images added, keypoints sceleton remained the same.
- [version 2](https://universe.roboflow.com/moorebot-scout/moorebot-pose-g9lqr/dataset/2): Initial small datset, sceleton definition updated, results more promisin.
- [version 1](https://universe.roboflow.com/moorebot-scout/moorebot-pose-g9lqr/dataset/1): Initial small dataset, keypoints sceleton very strict defined, results were not so encouraging.

## Dependencies

- `ultralytics` - YOLO implementation
- `opencv-python` - Image processing and display
- `torch` - PyTorch deep learning framework
- `torchvision` - Computer vision utilities

