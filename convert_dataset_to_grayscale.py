#!/usr/bin/env python3
"""
Convert a dataset split's images to grayscale (3-channel) and copy labels.
Writes a new data.yaml (optional) that points test/train/val to the new folders.

Usage:
    python convert_dataset_to_grayscale.py --dataset ./dataset/moorebot_v2 --split test --dest-name test_gray --update-yaml

This will create:
    ./dataset/moorebot_v2/test_gray/images/   (grayscale images, saved as 3-channel BGR)
    ./dataset/moorebot_v2/test_gray/labels/   (copied label .txt files)

If --update-yaml is passed, a new `data_gray.yaml` will be created alongside the original `data.yaml`.
"""

import argparse
import os
import shutil
from pathlib import Path
import cv2
import yaml


def convert_split_to_gray(dataset_root: Path, split: str, dest_name: str):
    src_images = dataset_root / split / 'images'
    src_labels = dataset_root / split / 'labels'
    if not src_images.exists():
        raise FileNotFoundError(f"Source images folder not found: {src_images}")

    dest_root = dataset_root / dest_name
    dest_images = dest_root / 'images'
    dest_labels = dest_root / 'labels'
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    images = [p for p in src_images.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    print(f"Converting {len(images)} images from {src_images} -> {dest_images}")

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  warning: failed to read {img_path}, skipping")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # convert back to 3-channel BGR so model input shape remains (H,W,3)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        dest_path = dest_images / img_path.name
        cv2.imwrite(str(dest_path), gray_bgr)

        # copy corresponding label file if present
        label_src = src_labels / (img_path.stem + '.txt')
        label_dest = dest_labels / (img_path.stem + '.txt')
        if label_src.exists():
            shutil.copy2(str(label_src), str(label_dest))

    print('Done converting images and copying labels.')
    return dest_root


def create_gray_data_yaml(orig_yaml: Path, dataset_root: Path, dest_name: str, out_yaml: Path):
    with open(orig_yaml, 'r') as f:
        data = yaml.safe_load(f)

    # Update test/train/val paths if they match default relative layout
    for key in ('train', 'val', 'test'):
        if key in data and isinstance(data[key], str):
            # replace only the split part; e.g. ../test/images -> ../test_gray/images
            parts = Path(data[key])
            # try to keep the same parent structure but substitute the split name when it matches
            # if path contains the original split name, replace it
            if parts.parts and parts.parts[-3:] and len(parts.parts) >= 3:
                # best effort: replace the last directory before images with dest_name when it's the original split
                data[key] = str(Path(parts.parent.parent) / dest_name / 'images') if parts.name == 'images' else str(parts)

    with open(out_yaml, 'w') as f:
        yaml.safe_dump(data, f)

    print(f'Wrote new data yaml: {out_yaml}')
    return out_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset root (contains data.yaml)')
    parser.add_argument('--split', type=str, default='test', help='Which split to convert (train/valid/test)')
    parser.add_argument('--dest-name', type=str, default=None, help='Destination split name (defaults to <split>_gray)')
    parser.add_argument('--update-yaml', action='store_true', help='Create a new data YAML (data_gray.yaml) pointing to the grayscale split')
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    if args.dest_name:
        dest_name = args.dest_name
    else:
        dest_name = f"{args.split}_gray"

    # Convert
    try:
        dest_root = convert_split_to_gray(dataset_root, args.split, dest_name)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Optionally create a new data yaml
    if args.update_yaml:
        orig_yaml = dataset_root / 'data.yaml'
        if not orig_yaml.exists():
            print(f"Original data.yaml not found at {orig_yaml}, cannot create updated yaml.")
            return
        out_yaml = dataset_root / 'data_gray.yaml'
        create_gray_data_yaml(orig_yaml, dataset_root, dest_name, out_yaml)
        print(f"Run evaluation with the new yaml: {out_yaml}")
    else:
        print(f"You can now point your evaluation to images in: {dest_root}")


if __name__ == '__main__':
    main()
