# Ad-hoc Inference Guide

This document covers running pose inference on arbitrary photos using the `run_inference.py` utility.

## Overview

The `run_inference.py` script allows you to run pose detection on any images that are not part of the training dataset structure. It supports single images, directories, or glob patterns.

## Basic Usage

### Simple Examples

```bash
# Single image
python run_inference.py --source misc/DSC04919.JPG --model runs/moorebot_v5/train-v1/yolo11n-moorebot_v5-pose-v1/weights/best.pt

# All JPG images inside a folder
python run_inference.py --source /path/to/new_photos --model runs/moorebot_v5/train-v1/yolo11n-moorebot_v5-pose-v1/weights/best.pt

# Glob pattern (quote globs so the shell doesn't expand unexpectedly)
python run_inference.py --source 'new_photos/**/*.png' --model runs/moorebot_v5/train-v1/yolo11n-moorebot_v5-pose-v1/weights/best.pt
```

### Advanced Examples

```bash
# Save JSON with structured detections (bboxes + keypoints)
python run_inference.py --source misc --save-json --out inference_outputs/moorebot_samples \
    --model runs/moorebot_v5/train-v1/yolo11n-moorebot_v5-pose-v1/weights/best.pt

# Grayscale (in-memory conversion) + show annotated window
python run_inference.py --source misc --grayscale --show \
    --model runs/moorebot_v5/train-v1/yolo11n-moorebot_v5-pose-v1/weights/best.pt

# Use each image's native resolution (no uniform square resize)
python run_inference.py --source misc --native \
    --model runs/moorebot_v5/train-v1/yolo11n-moorebot_v5-pose-v1/weights/best.pt

# Calculate robot orientation from keypoint triangle (assumes keypoint 0 is front)
python run_inference.py --source misc --calc-orientation \
    --model runs/moorebot_v5/train-v1/yolo11n-moorebot_v5-pose-v1/weights/best.pt

# Use keypoint 2 as front point for orientation calculation
python run_inference.py --source misc --calc-orientation --front-point 2 \
    --model runs/moorebot_v5/train-v1/yolo11n-moorebot_v5-pose-v1/weights/best.pt
```

## Command Line Options

### Core Options
- `--source`: Image file, directory, or glob pattern
- `--model`: Path to YOLO pose model .pt file (default: moorebot_v5 best model)
- `--out`: Output directory (default: auto timestamp under `inference_outputs/`)

### Image Processing
- `--imgsz`: Override inference size (default 640). Increase (e.g. 1280) for potentially higher accuracy at cost of speed
- `--native`: Preserve each image's original (H,W) resolution (model still internally pads to stride). Use for maximum spatial fidelity; expect slower & variable latency on large images
- `--grayscale`: Converts each image to grayscale before inference (useful for robustness checks)

### Detection Filtering
- `--conf`: Confidence threshold (default 0.25)
- `--iou`: IOU threshold for NMS (default 0.45)
- `--max`: Limit number of images processed (0 = all)

### Output Options
- `--save-json`: Exports a structured JSON (`results.json`) with bbox + keypoints per detection
- `--json-name`: Filename for JSON output (default: 'results.json')
- `--show`: Open a window displaying each annotated image (ESC to abort early)

### Orientation Calculation
- `--calc-orientation`: Calculate robot orientation in degrees based on triangle formed by 3 keypoints. Adds orientation data to JSON output
- `--front-point`: Which keypoint index (0, 1, or 2) represents the robot's front direction (default: 0). Only used with `--calc-orientation`

### System Options
- `--device`: Device override (e.g. 0, cpu)
- `--verbose`: Verbose model output

## JSON Output Schema

When using `--save-json`, the output follows this structure:

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
            ],
            "orientation": {
               "orientation_degrees": 53.0,
               "front_point_idx": 0,
               "front_point": {"x": 123.4, "y": 456.7},
               "centroid": {"x": 141.5, "y": 461.2},
               "triangle_area": 245.8,
               "keypoint_distances": [22.1, 18.5, 25.3],
               "avg_distance": 21.9
            }
         }
      ]
   }
]
```

**Note**: The `orientation` object is only included when using `--calc-orientation`.

## Robot Orientation Calculation

When using `--calc-orientation`, the script calculates the robot's orientation based on the triangle formed by the three keypoints:

### How It Works

1. **Triangle Centroid**: Calculated as the average of the three keypoint coordinates
2. **Front Vector**: Vector from centroid to the designated front keypoint (specified by `--front-point`)
3. **Orientation Angle**: Calculated using `atan2()` in degrees (0° = right, 90° = up, 180° = left, 270° = down)
4. **Additional Metrics**: Triangle area, distances from centroid to each keypoint, and average distance

### Applications

The orientation calculation is useful for:
- Robot navigation and path planning
- Automated tracking and following
- Quality control in manufacturing
- Directional analysis in research

### Important Notes

- Ensure you know which keypoint represents the robot's front direction
- This may vary depending on your dataset annotation convention
- Minimum confidence threshold of 0.3 is required for reliable orientation calculation
- All three keypoints must meet the confidence threshold

## Output Structure

Outputs are written as annotated images plus optional JSON to the chosen output directory:

- **Annotated Images**: Original filename with pose keypoints overlaid
- **JSON File**: Structured detection data (when `--save-json` is used)
- **Directory Structure**: Organized by timestamp or custom `--out` parameter

## Performance Considerations

### Image Size vs Speed
- **640x640** (default): Good balance of speed and accuracy
- **960x960**: Higher accuracy for small details, moderate speed impact
- **--native**: Maximum spatial fidelity, variable performance depending on source image size

### Memory Usage
- Large images with `--native` may require significant GPU memory
- Consider `--max` parameter to limit batch size for memory-constrained systems
- Grayscale conversion (`--grayscale`) reduces memory usage slightly

### Batch Processing
- Directory processing automatically handles batches
- Use `--max` to limit processing for testing
- Progress is shown for each image processed