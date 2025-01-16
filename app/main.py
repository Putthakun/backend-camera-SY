from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import base64

app = FastAPI()

# โหลด Haar Cascade สำหรับตรวจจับใบหน้า
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@app.websocket("/video_feed")
async def video_feed(websocket: WebSocket):
    await websocket.accept()

    # เปิดกล้อง (ใช้ index 0 สำหรับกล้องหลัก)
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # แปลงภาพเป็น Grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ตรวจจับใบหน้า
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

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
