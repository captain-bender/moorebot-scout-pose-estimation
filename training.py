from ultralytics import YOLO
import cv2
import argparse
import os


# Defaults
DEFAULT_DATA = "./dataset/moorebot_v2/data.yaml"
DEFAULT_PROJECT = "runs/moorebot_v2/train"
DEFAULT_NAME = "yolo11n-moorebot_v2-pose"


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO pose model with configurable paths")
    parser.add_argument("--data", type=str, default=os.environ.get("YOLO_DATA", DEFAULT_DATA),
                        help=f"path to data.yaml (default: {DEFAULT_DATA})")
    parser.add_argument("--project", type=str, default=os.environ.get("YOLO_PROJECT", DEFAULT_PROJECT),
                        help=f"project directory to save runs (default: {DEFAULT_PROJECT})")
    parser.add_argument("--name", type=str, default=os.environ.get("YOLO_RUN_NAME", DEFAULT_NAME),
                        help=f"run name (default: {DEFAULT_NAME})")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Training with data: {args.data}")
    print(f"Project dir: {args.project}")
    print(f"Run name: {args.name}")

    model = YOLO('yolo11n-pose.pt')

    # Train the model
    results = model.train(
        data=args.data,
        epochs=100,
        batch=5,
        workers=8,
        device=[0],
        imgsz=1280,
        amp=True,
        cache=True,
        project=args.project,
        name=args.name,
    )

    print(results)


if __name__ == "__main__":
    main()