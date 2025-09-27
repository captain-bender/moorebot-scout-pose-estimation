#!/usr/bin/env python3
"""
run_inference.py

Lightweight utility to run pose inference with a (pre)trained YOLO pose model on arbitrary
image files or folders of new photos (not necessarily in the training dataset structure).

Features:
- Accepts a single image, a directory, or a glob pattern (e.g. 'photos/*.jpg')
- Saves annotated images with keypoints overlaid
- Optional JSON export of detections (keypoints + confidences + bounding boxes)
- Optional grayscale conversion before inference (in-memory only)
- Adjustable confidence & IOU thresholds
- Optional display window for quick visual check
- Batch-friendly (no GUI) by default

Examples:
  # Single image, defaults
  python run_inference.py --source myphoto.jpg --model yolo11n-pose.pt

  # All jpg images in a folder, custom output dir, JSON export
  python run_inference.py --source /path/to/folder \
      --model runs/moorebot_v5/train-v1/yolo11n-moorebot_v5-pose-v1/weights/best.pt \
      --save-json --out outputs/moorebot_new_photos

  # Glob pattern with grayscale & show window
  python run_inference.py --source 'new_photos/**/*.png' --grayscale --show

JSON schema (list of detections per image):
[
  {
    "image": "filename.jpg",
    "width": 1920,
    "height": 1080,
    "detections": [
       {
         "bbox": [x1, y1, x2, y2],
         "score": 0.87,
         "keypoints": [
             {"x": 123.4, "y": 456.7, "conf": 0.91}, ...
         ]
       }, ...
    ]
  }, ...
]

Requirements:
  pip install ultralytics opencv-python torch torchvision

"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Sequence
import cv2

try:
    from ultralytics import YOLO
except ImportError as e:
    print("ERROR: ultralytics not installed. Install with: pip install ultralytics")
    raise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run YOLO pose model inference on new photos.")
    p.add_argument('--model', type=str, default='runs/moorebot_v5/train-v1/yolo11n-moorebot_v5-pose-v1/weights/best.pt', help='Path to YOLO pose model .pt file')
    p.add_argument('--source', type=str, required=True, help='Image file, directory, or glob pattern')
    p.add_argument('--out', type=str, default=None, help='Output directory (default: inference_outputs/<timestamp>)')
    p.add_argument('--imgsz', type=int, default=640, help='Inference image size (square). Ignored if --native is set.')
    p.add_argument('--native', action='store_true', help='Use each image\'s native resolution (no uniform resize)')
    p.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    p.add_argument('--iou', type=float, default=0.45, help='IOU threshold for NMS')
    p.add_argument('--max', type=int, default=0, help='Optional max number of images to process (0 = all)')
    p.add_argument('--grayscale', action='store_true', help='Convert image to grayscale before inference')
    p.add_argument('--save-json', action='store_true', help='Save detections as JSON (per run)')
    p.add_argument('--json-name', type=str, default='results.json', help='Filename for JSON output (inside out dir)')
    p.add_argument('--show', action='store_true', help='Show annotated images in a window')
    p.add_argument('--device', type=str, default=[0], help='Device override (e.g. 0, cpu)')
    p.add_argument('--verbose', action='store_true', help='Verbose model output')
    return p.parse_args()


def collect_image_paths(source: str) -> List[Path]:
    p = Path(source)
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    # If path exists and is file
    if p.exists() and p.is_file():
        return [p]
    # If path exists and is directory
    if p.exists() and p.is_dir():
        return sorted([f for f in p.rglob('*') if f.suffix.lower() in image_exts])

    # Treat as glob pattern
    paths = [Path(x) for x in sorted(Path().glob(source)) if Path(x).suffix.lower() in image_exts]
    if paths:
        return paths

    print(f"WARNING: No images found for source: {source}")
    return []


def ensure_out_dir(out: str | None) -> Path:
    if out is None:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        out_dir = Path('inference_outputs') / timestamp
    else:
        out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def convert_to_gray(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Keep shape compatible (H,W,1) then broadcast
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def run_inference_on_image(model: YOLO, img_path: Path, args: argparse.Namespace):
    # Load image (BGR)
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  Skipping unreadable image: {img_path}")
        return None, None
    if args.grayscale:
        img = convert_to_gray(img)

    # Ultralytics YOLO can take numpy array directly
    # Decide target size
    if args.native:
        # Provide [h, w] to preserve native shape (Ultralytics will still pad to stride multiple internally).
        h, w = img.shape[:2]
        infer_size = [h, w]
    else:
        infer_size = args.imgsz

    results = model.predict(img, imgsz=infer_size, conf=args.conf, iou=args.iou, verbose=args.verbose, device=args.device)
    if not results:
        return img, []
    detections_serializable = []

    # Annotated frame (first result only, pose tasks usually one result per image)
    result = results[0]
    annotated = result.plot()

    # Extract detections: boxes + keypoints
    try:
        boxes = result.boxes
        kpts = result.keypoints
        if boxes is not None and len(boxes) > 0:
            import numpy as np
            import torch

            # Boxes tensor shape (n,6) -> xyxy + conf + cls (depending on version)
            # We'll use boxes.xyxy, boxes.conf
            xyxy = boxes.xyxy
            confs = getattr(boxes, 'conf', None)
            cls = getattr(boxes, 'cls', None)
            # Keypoints: kpts.xy shape (n,k,2), kpts.conf shape (n,k,1)
            kp_xy = kpts.xy if (kpts is not None and hasattr(kpts, 'xy')) else None
            kp_conf = kpts.conf if (kpts is not None and hasattr(kpts, 'conf')) else None

            # Convert to numpy
            if isinstance(xyxy, torch.Tensor):
                xyxy = xyxy.cpu().numpy()
            if confs is not None and isinstance(confs, torch.Tensor):
                confs = confs.cpu().numpy()
            if cls is not None and isinstance(cls, torch.Tensor):
                cls = cls.cpu().numpy()
            if kp_xy is not None and isinstance(kp_xy, torch.Tensor):
                kp_xy = kp_xy.cpu().numpy()
            if kp_conf is not None and isinstance(kp_conf, torch.Tensor):
                kp_conf = kp_conf.squeeze(-1).cpu().numpy()

            n = xyxy.shape[0]
            for i in range(n):
                det = {
                    'bbox': [float(x) for x in xyxy[i].tolist()],
                    'score': float(confs[i]) if confs is not None else None,
                    'class': int(cls[i]) if cls is not None else None,
                }
                if kp_xy is not None:
                    pts = []
                    for k in range(kp_xy.shape[1]):
                        px, py = float(kp_xy[i, k, 0]), float(kp_xy[i, k, 1])
                        pc = float(kp_conf[i, k]) if (kp_conf is not None) else None
                        pts.append({'x': px, 'y': py, 'conf': pc})
                    det['keypoints'] = pts
                detections_serializable.append(det)
    except Exception as e:
        # Keep error on single line to avoid breaking f-string
        print(f"  Warning: failed to extract structured detections for {img_path}: {e} ({type(e).__name__})")

    return annotated, detections_serializable


def main():
    args = parse_args()

    print("\nRunning inference on new photos")
    print(f" Model: {args.model}")
    print(f" Source: {args.source}")
    if args.native:
        print(" Image sizing: native per-image resolution")
    else:
        print(f" Image sizing: fixed square {args.imgsz}")

    image_paths = collect_image_paths(args.source)
    if not image_paths:
        print("No images to process. Exiting.")
        sys.exit(1)

    if args.max > 0:
        image_paths = image_paths[:args.max]

    out_dir = ensure_out_dir(args.out)
    print(f" Output dir: {out_dir}")
    if args.grayscale:
        print(" Grayscale: enabled (in-memory conversion)")

    # Load model
    print(" Loading model...")
    model = YOLO(args.model)

    json_records = []

    for idx, img_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] {img_path.name}")
        annotated, detections = run_inference_on_image(model, img_path, args)
        if annotated is None:
            continue

        # Save annotated image
        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), annotated)

        # JSON record
        if args.save_json:
            h, w = annotated.shape[:2]
            json_records.append({
                'image': img_path.name,
                'width': w,
                'height': h,
                'detections': detections
            })

        if args.show:
            try:
                cv2.imshow('inference', annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC to exit early
                    print(" ESC pressed. Stopping early.")
                    break
            except Exception as e:
                print(f"  Warning: display failed: {e}")
                args.show = False  # Disable further attempts

    if args.show:
        cv2.destroyAllWindows()

    if args.save_json and json_records:
        json_path = out_dir / args.json_name
        with open(json_path, 'w') as f:
            json.dump(json_records, f, indent=2)
        print(f" JSON saved: {json_path}")

    print("\nInference complete.")
    print(f"Processed {len(image_paths)} image(s). Outputs in: {out_dir}")


if __name__ == '__main__':
    main()
