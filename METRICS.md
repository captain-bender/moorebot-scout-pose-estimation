# Model Evaluation and Metrics

This document covers the comprehensive evaluation metrics and testing procedures for the Moorebot pose detection model.

## Overview

The project includes several evaluation scripts that measure different aspects of model performance:

```bash
python test-evaluation.py      # Comprehensive test metrics
python test-individual.py      # Individual image testing
python pose-metrics-analysis.py # Detailed pose analysis
```

## Evaluation Metrics

### 1) Pose Estimation Metrics
Evaluates the accuracy of the predicted robot poses against ground truth (GT) annotations:
- **mAP50**: Mean Average Precision at IoU=0.50.
- **mAP75**: Mean Average Precision at IoU=0.75.
- **mAP50-95**: Mean Average Precision at IoU=0.50 to 0.95 (incremental 0.05).

**Notes:**
- Ensure your test data is correctly formatted and includes GT annotations.

### 2) PCK (Percentage of Correct Keypoints)
Measures the percentage of correctly predicted keypoints:
- Overall and per keypoint at thresholds 0.1 and 0.2.

**Notes:**
- Requires correct GT keypoint annotations in the test set.

### 3) OKS (Object Keypoint Similarity)
Evaluates the similarity between predicted and GT keypoints, normalized by object area:
- OKS AP@[.50:.95]: mean AP over thresholds 0.50, 0.55, …, 0.95.
- OKS AP@0.50: lenient match criterion.
- OKS AP@0.75: stricter localization.

**Notes:**
- OKS thresholds can be adjusted in the script for different evaluation criteria.

### 4) Inference Time
Measures the time taken for the model to predict poses on the test set.

**Notes:**
- Ensure a consistent environment for timing (e.g., same hardware, no other heavy processes running).

### 5) Keypoint Confidence
Aggregates confidences of predicted keypoints across the test set (no GT matching):
- Overall: count, mean, median, std, and share above thresholds (default: 0.3, 0.5, 0.7).
- Per-detection: average keypoint confidence per detected instance (mean/median).
- Per-keypoint: mean, count, and share above thresholds (e.g., >= 0.5) for each keypoint index (uses `KEYPOINT_NAMES`).

**Notes:**
- This summarizes predicted confidences only.

## Test Results Organization

Outputs from testing are saved under a structured `tests/` directory:

- **Pattern**: `tests/<dataset>/<split>/<model>/<tag|timestamp>/`
- **Example**: `tests/moorebot_v4/test/yolo11n-moorebot_v5-pose-v1/smoke/DSC04824...jpg`

Control the final folder with `--tag` (falls back to a timestamp if omitted):

```bash
python test-individual.py --count 2 --tag smoke
```

## Grayscale Testing

Evaluate model robustness to lighting and texture variations:

```bash
python test-individual.py --grayscale
```

**Details:**
- Temporary grayscale copies are created under a split-specific temp folder (e.g., `dataset/<name>/<split>_gray_tmp`)
- Files are automatically deleted at the end of the run
- To keep the grayscale temp images for inspection, add `--keep-temp`

## Cleanup Utilities

If any grayscale temp folders remain (e.g., after an interrupted run), use the cleanup utility:

```bash
# Dry run: show what would be removed
python3 cleanup_gray_tmp.py --dry-run

# Actually remove found *_gray_tmp directories (non-interactive)
python3 cleanup_gray_tmp.py --yes

# Optional: limit scan to a specific root
python3 cleanup_gray_tmp.py --root dataset --yes
```

## Performance Interpretation

### Speed Categories
- **Real-time (>30 FPS)**: < 33 ms per image
- **Near real-time (>15 FPS)**: < 67 ms per image  
- **Interactive (>10 FPS)**: < 100 ms per image
- **Batch processing**: > 100 ms per image

### Application Requirements
- **Robot monitoring**: Need >15 FPS (< 67ms)
- **Interactive robot apps**: Need >10 FPS (< 100ms)  
- **Industrial automation**: Need >30 FPS (< 33ms)
- **Quality control**: Speed less critical

### Optimization Tips
- Use smaller image sizes (640x640) for speed
- Use larger sizes (1280+) for accuracy
- Consider model quantization for production
- GPU is much faster than CPU
- Moorebot detection benefits from consistent lighting