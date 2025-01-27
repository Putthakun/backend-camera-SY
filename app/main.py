from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import base64
import asyncio
import time
import os

from app.rabbitmq import *

app = FastAPI()

# โหลด Haar Cascade สำหรับตรวจจับใบหน้า
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# การเชื่อมต่อ RabbitMQ ตลอดเวลา
connection, channel = get_rabbitmq_connection()

output_folder = "faces"
os.makedirs(output_folder, exist_ok=True)  # สร้างโฟลเดอร์ถ้ายังไม่มี


@app.websocket("/video_feed")
async def video_feed(websocket: WebSocket):

    await websocket.accept()

    # เปิดกล้อง (ใช้ index 0 สำหรับกล้องหลัก)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        # ส่งข้อความไปที่ frontend ถ้ากล้องไม่พร้อม
        await websocket.send_text("Unable to connect camera, please check the camera.")
        cap.release()
        return

    image_count = 0  # นับจำนวนภาพที่บันทึก

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await websocket.send_text("The camera is not turned on.")
                break

            # แปลงภาพเป็น Grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ตรวจจับใบหน้า
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
            for (x, y, w, h) in faces:
                # Crop ภาพใบหน้าตามค่าที่ตรวจจับ
                face_crop = frame[y:y+h, x:x+w]  

                # บันทึกภาพที่ถูก crop
                image_count += 1
                filename = os.path.join(output_folder, f"face_{image_count}.jpg")
                cv2.imwrite(filename, face_crop)
                print(f"บันทึกใบหน้าที่ {image_count}: {filename}")

            # วาดกรอบรอบใบหน้า
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # แปลงภาพเป็น base64 เพื่อส่งผ่าน WebSocket
            _, buffer = cv2.imencode(".jpg", frame)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")

            # ส่งภาพไปยัง Client
            await websocket.send_text(frame_base64)

    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        cap.release()
        await websocket.close()