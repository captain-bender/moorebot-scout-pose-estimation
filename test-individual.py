from ultralytics import YOLO
import cv2
import os
import argparse
from pathlib import Path
import shutil
import yaml
from datetime import datetime
import random


def make_gray_images_if_needed(dataset_root: Path, split: str, img_names: list[str], keep_temp: bool = False) -> Path:
    """Return path to temp split root containing grayscale copies of the provided image names.
    Converts only the requested images using convert_dataset_to_grayscale.convert_images_to_gray.
    """
    from convert_dataset_to_grayscale import convert_images_to_gray

    dest_name = f"{split}_gray_tmp"
    dest_root = convert_images_to_gray(dataset_root, split, dest_name, img_names)
    return dest_root


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='runs/moorebot_v5/train-v1/yolo11n-moorebot_v5-pose-v1/weights/best.pt')
    parser.add_argument('--dataset', type=str, default='./datasets/moorebot_v5')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--count', type=int, default=2, help='Number of images to run')
    parser.add_argument('--grayscale', action='store_true', help='Convert images to grayscale before inference')
    parser.add_argument('--keep-temp', action='store_true', help='Do not delete temporary grayscale images')
    parser.add_argument('--tag', type=str, default='', help='Optional tag to include in tests output path')
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)

    test_images_dir = Path(args.dataset) / args.split / 'images'

    print("Testing on individual images...")

    all_images = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not all_images:
        print(f"No images found in {test_images_dir}")
        return
    k = min(args.count, len(all_images))
    test_images = random.sample(all_images, k=k)

    # track temporary grayscale folder to optionally cleanup
    gray_dir = None

    try:
        if args.grayscale:
            print(f"Preparing grayscale images for: {test_images}")
            gray_root = make_gray_images_if_needed(Path(args.dataset), args.split, test_images, keep_temp=args.keep_temp)
            gray_dir = gray_root

        for img_name in test_images:
            img_path = (gray_dir / 'images' / img_name) if (args.grayscale and gray_dir is not None) else (Path(test_images_dir) / img_name)

            print(f"\n Processing: {img_name}")

            # Run inference
            results = model(str(img_path))

            # Access results
            for result in results:
                # Plot first so we can access the palette used for keypoints
                annotated_frame = result.plot()

                # Get keypoints
                if result.keypoints is not None:
                    xy = result.keypoints.xy  # x and y coordinates
                    conf = result.keypoints.conf  # confidence scores
                    print(f"   Detected {len(xy)} pose(s)")
                    print(f"   Keypoints shape: {xy.shape}")

                    try:
                        import torch
                        xy_np = xy.cpu().numpy() if isinstance(xy, torch.Tensor) else xy
                        conf_arr = None
                        if conf is not None:
                            conf_arr = conf
                            if isinstance(conf_arr, torch.Tensor):
                                conf_arr = conf_arr.squeeze(-1).cpu().numpy()

                        # Determine per-keypoint colors matching Ultralytics plotting
                        kp_colors = None
                        num_kpts = xy_np.shape[1]
                        try:
                            from ultralytics.utils.plotting import kpt_color as KP_COLORS, colors as ucolors
                            if KP_COLORS and len(KP_COLORS) > 0:
                                # Use provided keypoint palette; pad/trim to match current number of keypoints
                                kp_colors = [tuple(int(c) for c in KP_COLORS[i % len(KP_COLORS)]) for i in range(num_kpts)]
                            else:
                                kp_colors = [ucolors(i, True) for i in range(num_kpts)]
                        except Exception:
                            try:
                                from ultralytics.utils.plotting import colors as ucolors
                                kp_colors = [ucolors(i, True) for i in range(num_kpts)]
                            except Exception:
                                kp_colors = None

                        for i in range(xy_np.shape[0]):
                            print(f"   Pose {i} keypoints:")
                            for k in range(xy_np.shape[1]):
                                xk, yk = float(xy_np[i, k, 0]), float(xy_np[i, k, 1])
                                color_str = ""
                                if kp_colors is not None and k < len(kp_colors):
                                    col = kp_colors[k]
                                    # Ensure tuple of ints as BGR
                                    try:
                                        bgr = tuple(int(c) for c in col)
                                    except Exception:
                                        bgr = col
                                    color_str = f", color(BGR)={bgr}"
                                if conf_arr is not None:
                                    ck = float(conf_arr[i, k])
                                    print(f"     kp[{k}]: x={xk:.1f}, y={yk:.1f}, conf={ck:.3f}{color_str}")
                                else:
                                    print(f"     kp[{k}]: x={xk:.1f}, y={yk:.1f}{color_str}")
                    except Exception as e:
                        print(f"   Warning: failed to print per-keypoint confidences/colors: {e}")
                # Build structured output directory under tests/
                dataset_name = Path(args.dataset).name
                model_name = Path(args.model).parent.parent.name if 'weights' in args.model else Path(args.model).stem
                timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
                tag = args.tag.strip().replace(' ', '_')
                base_dir = Path('tests') / dataset_name / args.split / model_name
                out_dir = base_dir / (tag if tag else timestamp)
                out_dir.mkdir(parents=True, exist_ok=True)

                output_path = out_dir / f"{img_name}"
                cv2.imwrite(str(output_path), annotated_frame)
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
