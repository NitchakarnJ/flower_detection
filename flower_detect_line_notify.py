




import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import platform
import pathlib
import torch
import cv2
import requests  

# Line Notify token
LINE_NOTIFY_TOKEN = "" 

def send_line_notify_with_image(message, image_path):
    """ส่งข้อความพร้อมรูปภาพผ่าน Line Notify"""
    url = "https://notify-api.line.me/api/notify"
    headers = {
        "Authorization": f"Bearer {LINE_NOTIFY_TOKEN}"
    }
    data = {
        "message": message
    }
    files = {
        "imageFile": open(image_path, "rb")
    }
    response = requests.post(url, headers=headers, data=data, files=files)
    if response.status_code != 200:
        print("Failed to send Line Notify:", response.text)

# Fix pathlib compatibility
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# Load custom weights
weights_path = os.path.join(os.path.dirname(__file__), 'best_10.pt')
if not os.path.exists(weights_path):
    print(f"Error: Weights file not found at {weights_path}")
    exit()

# Load the YOLOv5 model
model = torch.hub.load('./yolov5', "custom", path=weights_path, source="local")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.60  # Adjust as needed

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Perform inference
    results = model(frame)

    # Filter results based on confidence threshold
    detections = results.pandas().xyxy[0]  # Get detections as a DataFrame
    detections = detections[detections['confidence'] > CONFIDENCE_THRESHOLD]

    # Draw bounding boxes and labels manually
    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']
        label = row['name']

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put label and confidence
        cv2.putText(
            frame, 
            f"{label} {confidence:.2f}", 
            (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 0), 
            2
        )

        # Save the frame with detections
        detected_image_path = "detected_object.jpg"
        cv2.imwrite(detected_image_path, frame)

        # ส่งแจ้งเตือนพร้อมรูปภาพผ่าน Line Notify
        send_line_notify_with_image(
            f"Detected: {label} with confidence {confidence:.2f}", 
            detected_image_path
        )

    # Show frame with detections
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
