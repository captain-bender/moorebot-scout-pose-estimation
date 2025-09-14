from ultralytics import YOLO
import cv2
import os
import argparse
from pathlib import Path
import shutil
import yaml


def make_gray_image_if_needed(dataset_root: Path, split: str, img_name: str, keep_temp: bool = False) -> Path:
    """Return path to image to use for inference.
    If grayscale is requested, create a grayscale copy for this image under a temp split and return its path.
    """
    from convert_dataset_to_grayscale import convert_split_to_gray

    dest_name = f"{split}_gray_tmp"
    dest_root = convert_split_to_gray(dataset_root, split, dest_name)
    img_path = dest_root / 'images' / img_name
    return img_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='runs/moorebot_v4/train-v1/yolo11n-moorebot_v4-pose-v1/weights/best.pt')
    parser.add_argument('--dataset', type=str, default='./dataset/moorebot_v4')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--count', type=int, default=5, help='Number of images to run')
    parser.add_argument('--grayscale', action='store_true', help='Convert images to grayscale before inference')
    parser.add_argument('--keep-temp', action='store_true', help='Do not delete temporary grayscale images')
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)

    test_images_dir = Path(args.dataset) / args.split / 'images'

    print("Testing on individual images...")

    test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:args.count]

    # track temporary grayscale folder to optionally cleanup
    gray_dir = None

    try:
        for img_name in test_images:
            if args.grayscale:
                print(f"Preparing grayscale image for: {img_name}")
                img_path = make_gray_image_if_needed(Path(args.dataset), args.split, img_name, keep_temp=args.keep_temp)
                gray_dir = Path(args.dataset) / f"{args.split}_gray_tmp"
            else:
                img_path = Path(test_images_dir) / img_name

            print(f"\n Processing: {img_name}")

            # Run inference
            results = model(str(img_path))

            # Access results
            for result in results:
                # Get keypoints
                if result.keypoints is not None:
                    xy = result.keypoints.xy  # x and y coordinates
                    conf = result.keypoints.conf  # confidence scores
                    print(f"   Detected {len(xy)} pose(s)")
                    print(f"   Keypoints shape: {xy.shape}")

                # Plot and save results
                annotated_frame = result.plot()
                output_path = f"test_results_{img_name}"
                cv2.imwrite(output_path, annotated_frame)
                print(f" Saved result: {output_path}")

    finally:
        # cleanup grayscale temp if created and not requested to keep
        if args.grayscale and gray_dir is not None and not args.keep_temp:
            try:
                print(f"Removing temporary grayscale folder: {gray_dir}")
                shutil.rmtree(gray_dir)
            except Exception as e:
                print(f"Warning: failed to remove temp grayscale folder: {e}")

    print("\n Individual image testing complete!")


if __name__ == '__main__':
    main()
