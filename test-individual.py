from ultralytics import YOLO
import cv2
import os

# Load your trained model
model = YOLO('runs/moorebot_v2/train/yolo11n-moorebot_v2-pose/weights/best.pt')

# Test on individual images from test set
test_images_dir = "./dataset/moorebot_v2/test/images"

print("Testing on individual images...")

# Get a few test images
test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:5]

for img_name in test_images:
    img_path = os.path.join(test_images_dir, img_name)
    
    print(f"\n Processing: {img_name}")
    
    # Run inference
    results = model(img_path)
    
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

print("\n Individual image testing complete!")
