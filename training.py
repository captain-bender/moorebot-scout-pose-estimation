from ultralytics import YOLO
import cv2

model = YOLO('yolo11n-pose.pt')

# Train the model
results = model.train(
    # data="./dataset/glue/data.yaml", 
    data="./dataset/moorebot_v2/data.yaml",
    epochs=100,
    batch=5,  # Reduced from 8 to 2 to save memory
    imgsz=1980,  # Reduced from 1920 to 1280 to save memory
    #project="runs/train",
    project="runs/moorebot_v2/train",
    #name="yolo11n-glue-pose"
    name="yolo11n-moorebot_v2-pose"
)

print(results)