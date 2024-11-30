import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from flask import Flask, jsonify
import os
import platform
import pathlib
import torch
import cv2
import requests  # สำหรับส่งข้อความผ่าน Line Notify

# สร้าง Flask app
app = Flask(__name__)

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath


# โหลด YOLOv5
weights_path = "best_10.pt"  # ใส่ path ที่ถูกต้องของไฟล์ weights
model = torch.hub.load('./yolov5', 'custom', path=weights_path, source='local')

# เปิดกล้อง (Webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# ฟังก์ชันในการส่งข้อมูลผ่าน Line Notify
LINE_NOTIFY_TOKEN = "5csNQjR898Iev5G5ENGXP3q7QmNr6l5ZDZBvjHhwLhf"
def send_line_notify(message):
    url = "https://notify-api.line.me/api/notify"
    headers = {
        "Authorization": f"Bearer {LINE_NOTIFY_TOKEN}"
    }
    data = {"message": message}
    requests.post(url, headers=headers, data=data)

# API สำหรับให้ข้อมูลการตรวจจับจาก YOLOv5
@app.route('/api/detect', methods=['GET'])
def detect_objects():
    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Failed to capture frame"}), 500

    # ทำการตรวจจับด้วย YOLO
    results = model(frame)
    detections = results.pandas().xyxy[0]  # รับข้อมูลการตรวจจับในรูปแบบ DataFrame

    detected_objects = []
    for _, row in detections.iterrows():
        if row['confidence'] > 0.60:  # กรองเฉพาะผลที่มีความมั่นใจมากกว่า 60%
            detected_objects.append({
                "label": row['name'],
                "confidence": row['confidence']
            })

            # ส่งข้อความแจ้งเตือนผ่าน Line Notify
            send_line_notify(f"Detected: {row['name']} with confidence {row['confidence']:.2f}")

    return jsonify({"objects": detected_objects})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
