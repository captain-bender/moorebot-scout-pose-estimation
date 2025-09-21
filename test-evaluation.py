from ultralytics import YOLO
import cv2
import argparse
import tempfile
import shutil
from pathlib import Path
import yaml
import subprocess
import time


def make_gray_split_if_needed(dataset_root: Path, split: str):
    """Create a grayscale copy of the split images and return (tmp_yaml_path, dest_root_path)."""
    from convert_dataset_to_grayscale import convert_split_to_gray

    dest_name = f"{split}_gray_tmp"
    dest_root = convert_split_to_gray(dataset_root, split, dest_name)

    # Create a temporary data yaml referencing the new gray split for evaluation
    orig_yaml = dataset_root / 'data.yaml'
    if not orig_yaml.exists():
        raise FileNotFoundError(f"Original data.yaml not found at {orig_yaml}")

    with open(orig_yaml, 'r') as f:
        data = yaml.safe_load(f)

    # Update the test path to the gray images
    data['test'] = str(dest_root / 'images')

    tmp_yaml = dataset_root / 'data_gray_tmp.yaml'
    with open(tmp_yaml, 'w') as f:
        yaml.safe_dump(data, f)

    return tmp_yaml, dest_root


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='runs/moorebot_v5/train-v1/yolo11n-moorebot_v5-pose-v1/weights/best.pt',
                        help='Path to trained model .pt')
    parser.add_argument('--data', type=str, default='./datasets/moorebot_v5/data.yaml', help='Path to data.yaml')
    parser.add_argument('--grayscale', action='store_true', help='Evaluate on a grayscale copy of the test split')
    parser.add_argument('--keep-temp', action='store_true', help='Do not delete temporary grayscale data and yaml')
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)

    print("Evaluating model on test dataset...")

    data_yaml = Path(args.data)
    tmp_yaml = None
    gray_dir = None
    if args.grayscale:
        dataset_root = data_yaml.parent
        print("Creating grayscale test split (temporary)...")
        tmp_yaml, gray_dir = make_gray_split_if_needed(dataset_root, 'test')
        data_to_use = str(tmp_yaml)
    else:
        data_to_use = str(data_yaml)

    # Evaluate on test set (ensure cleanup in finally)
    results = None
    try:
        results = model.val(
            data=data_to_use,
            split='test',
            imgsz=1280,
            batch=4,
            save_json=True,
            save_hybrid=True,
            plots=True,
            verbose=True
        )
    finally:
        # Cleanup temporary yaml and grayscale folder if created, unless user asked to keep temps
        if args.grayscale and not args.keep_temp:
            try:
                if tmp_yaml is not None and tmp_yaml.exists():
                    print(f"Removing temporary yaml: {tmp_yaml}")
                    tmp_yaml.unlink()
                if gray_dir is not None and gray_dir.exists():
                    print(f"Removing temporary grayscale folder: {gray_dir}")
                    shutil.rmtree(gray_dir)
            except Exception as e:
                print(f"Warning: failed to clean up temporary files: {e}")

    if results is None:
        print("Evaluation failed or was interrupted; no results to show.")
        return

    print("\n Test Results:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")

    # If you have pose metrics
    if hasattr(results, 'pose'):
        print(f"Pose mAP50: {results.pose.map50:.4f}")
        print(f"Pose mAP50-95: {results.pose.map:.4f}")

    # Get save directory from results
    if hasattr(results, 'save_dir'):
        print(f"\n Results saved to: {results.save_dir}")
    else:
        print(f"\n Results saved to: runs/detect/ (default location)")

    # Per-image timing and throughput (Hz)
    try:
        data_yaml_path = Path(data_to_use)
        with open(data_yaml_path, 'r') as f:
            data_cfg = yaml.safe_load(f)
        test_entry = data_cfg.get('test')
        if test_entry is None:
            print("No 'test' entry found in data yaml; skipping per-image timing.")
        else:
            # Resolve the test images directory robustly
            raw = str(test_entry)
            candidates = []
            # As given (CWD relative or absolute)
            try:
                candidates.append(Path(raw).resolve())
            except Exception:
                pass
            # Relative to YAML parent
            try:
                candidates.append((data_yaml_path.parent / raw).resolve())
            except Exception:
                pass
            # If YAML defines a 'path' root, try relative to it
            ds_root = data_cfg.get('path')
            if ds_root:
                try:
                    candidates.append((Path(ds_root) / raw).resolve())
                except Exception:
                    pass
            # If starts with ./ or ../, also try stripped version
            if raw.startswith('../') or raw.startswith('./'):
                stripped = raw.lstrip('./')
                try:
                    candidates.append((data_yaml_path.parent / stripped).resolve())
                except Exception:
                    pass
                if ds_root:
                    try:
                        candidates.append((Path(ds_root) / stripped).resolve())
                    except Exception:
                        pass
                try:
                    candidates.append(Path(stripped).resolve())
                except Exception:
                    pass

            # Pick the first existing dir
            test_path = None
            for c in candidates:
                if isinstance(c, Path) and c.is_dir():
                    test_path = c
                    break

            if test_path and test_path.is_dir():
                # Collect images
                img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
                images = sorted([p for p in test_path.iterdir() if p.suffix.lower() in img_exts])
                if not images:
                    print(f"No images found under {test_path}; skipping per-image timing.")
                else:
                    print("\n Per-image timing (single-image inference):")
                    times = []
                    for img in images:
                        t0 = time.perf_counter()
                        _ = model(str(img), imgsz=1280)
                        t1 = time.perf_counter()
                        dt = t1 - t0
                        hz = (1.0 / dt) if dt > 0 else float('inf')
                        times.append(dt)
                        print(f"  {img.name}: {dt*1000:.1f} ms | {hz:.2f} Hz")

                    # Summary
                    if times:
                        n = len(times)
                        total = sum(times)
                        avg = total / n
                        mn = min(times)
                        mx = max(times)
                        print("\n Timing summary:")
                        print(f"  Images: {n}")
                        print(f"  Avg: {avg*1000:.1f} ms | {1.0/avg if avg>0 else float('inf'):.2f} Hz")
                        print(f"  Min: {mn*1000:.1f} ms | {1.0/mn if mn>0 else float('inf'):.2f} Hz")
                        print(f"  Max: {mx*1000:.1f} ms | {1.0/mx if mx>0 else float('inf'):.2f} Hz")
            else:
                print("Unable to resolve a valid 'test' images directory; skipping per-image timing.")
    except Exception as e:
        print(f"Warning: failed per-image timing: {e}")

    # Cleanup temporary yaml and grayscale folder if created
    if tmp_yaml is not None:
        try:
            tmp_root = tmp_yaml.parent
            print(f"Cleaning up temporary files: {tmp_yaml}")
            # Remove the grayscale split folder
            gray_dir = tmp_root / 'test_gray_tmp'
            if gray_dir.exists():
                shutil.rmtree(gray_dir)
            tmp_yaml.unlink()
        except Exception as e:
            print(f"Warning: failed to clean up temporary files: {e}")


if __name__ == '__main__':
    main()
