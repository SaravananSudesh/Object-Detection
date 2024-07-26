import torch
import cv2
import numpy as np
import os

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load the video
cap = cv2.VideoCapture('input_video.mp4')

output_dir = 'objects'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frame_count = 0
crop_count = 0
saved_crops = []

def is_duplicate(new_crop, saved_crops):
    """Check if the new_crop is a duplicate of any saved crops."""
    for crop in saved_crops:
        if np.array_equal(new_crop, crop):
            return True
    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    for i, det in enumerate(results.xyxy[0]):  # x1, y1, x2, y2, confidence, class
        x1, y1, x2, y2, conf, cls = map(int, det)
        cropped_image = frame[y1:y2, x1:x2]

        # Check for duplicates
        if not is_duplicate(cropped_image, saved_crops):
            saved_crops.append(cropped_image)
            crop_filename = os.path.join(output_dir, f'frame_{frame_count}_crop_{i}.jpg')
            cv2.imwrite(crop_filename, cropped_image)
            crop_count += 1

    frame_count += 1

cap.release()
cv2.destroyAllWindows()