import cv2
import os
import numpy as np

def parse_yolo_annotation(anno_path, img_width, img_height):
    """
    Parses a YOLO-formatted annotation file and returns bounding boxes and keypoints.
    """
    bboxes = []
    keypoints = []
    
    if not os.path.exists(anno_path):
        return bboxes, keypoints

    with open(anno_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
            
        class_id = int(parts[0])
        # Bounding Box (normalized to pixel values)
        x_center, y_center, w, h = [float(p) for p in parts[1:5]]
        x_min = int((x_center - w/2) * img_width)
        y_min = int((y_center - h/2) * img_height)
        x_max = int((x_center + w/2) * img_width)
        y_max = int((y_center + h/2) * img_height)
        bboxes.append({'class_id': class_id, 'bbox': [x_min, y_min, x_max, y_max]})

        # Keypoints
        if len(parts) > 5:
            keypoint_data = [float(p) for p in parts[5:]]
            for i in range(0, len(keypoint_data), 3):
                x_kp = int(keypoint_data[i] * img_width)
                y_kp = int(keypoint_data[i+1] * img_height)
                visibility = int(keypoint_data[i+2])
                keypoints.append({'class_id': class_id, 'kp': [x_kp, y_kp], 'visibility': visibility})

    return bboxes, keypoints

def main():
    """
    Main function to iterate through images with keyboard navigation.
    """
    # Define your data paths
    images_dir = './dataset/train/images/'
    annotations_dir = './dataset/train/labels/'

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('jpg', 'jpeg', 'png'))])

    if not image_files:
        print("No images found in the specified directory.")
        return

    current_image_index = 0
    cv2.namedWindow('Image Annotations', cv2.WINDOW_NORMAL)
    
    while True:
        current_image_name = image_files[current_image_index]
        image_path = os.path.join(images_dir, current_image_name)
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}. Skipping.")
            current_image_index = (current_image_index + 1) % len(image_files)
            continue
            
        img_h, img_w, _ = img.shape
        base_name = os.path.splitext(current_image_name)[0]
        annotation_path = os.path.join(annotations_dir, base_name + '.txt')
        annotation_file_name = os.path.basename(annotation_path)

        bboxes, keypoints = parse_yolo_annotation(annotation_path, img_w, img_h)

        # Draw bounding boxes
        for box_data in bboxes:
            x_min, y_min, x_max, y_max = box_data['bbox']
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
        # Draw keypoints
        for kp_data in keypoints:
            x, y = kp_data['kp']
            visibility = kp_data['visibility']
            if visibility > 0:
                color = (0, 0, 255) if visibility == 2 else (255, 0, 0)
                cv2.circle(img, (x, y), 5, color, -1)
        
        # Add file names to the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_color = (255, 255, 255) # White color
        
        # Display image file name
        img_name_text = f"Image: {current_image_name}"
        cv2.putText(img, img_name_text, (10, 20), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
        # Display annotation file name
        anno_name_text = f"Annotation: {annotation_file_name}"
        cv2.putText(img, anno_name_text, (10, 40), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
        display_img = cv2.resize(img, (800, 600))
        cv2.imshow('Image Annotations', display_img)
        
        print(f"Displaying {current_image_name}. Use arrow keys to navigate. ESC to exit.")
        
        key = cv2.waitKey(0)

        if key == 27:
            break
        elif key == 83:  # Right arrow
            current_image_index = (current_image_index + 1) % len(image_files)
        elif key == 81:  # Left arrow
            current_image_index = (current_image_index - 1 + len(image_files)) % len(image_files)
        
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()