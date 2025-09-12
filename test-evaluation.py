from ultralytics import YOLO
import cv2

# Load your trained model
# Use 'best.pt' for the best performing model during training
model = YOLO('runs/moorebot_v2/train/yolo11n-moorebot_v2-pose/weights/best.pt')

# Alternative: use 'last.pt' for the final epoch model
# model = YOLO('runs/moorebot/train/yolo11n-moorebot-pose/weights/last.pt')

print("Evaluating model on test dataset...")

# Evaluate on test set
results = model.val(
    data="./dataset/moorebot_v2/data.yaml",
    split='test',  # Use test split instead of val split
    imgsz=1280,    # Same image size used during training
    batch=4,       # Same batch size as training
    save_json=True,  # Save results in COCO format
    save_hybrid=True,  # Save hybrid labels (useful for analysis)
    plots=True,    # Generate evaluation plots
    verbose=True   # Detailed output
)

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
