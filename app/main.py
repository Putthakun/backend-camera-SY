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

camera_id = "CAM_01"

def compress_and_encode_image(image, quality=100):
    """ บีบอัดภาพแล้วเข้ารหัสเป็น bytes """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image ต้องเป็น numpy array")

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    
    # ลองเช็คว่าภาพที่รับเข้ามาเป็นแบบไหน
    print(f"📌 Shape of image: {image.shape} | Dtype: {image.dtype}")

    success, encoded_image = cv2.imencode(".jpg", image, encode_param)
    
    if success:
        print(f"✅ Encoded image size: {len(encoded_image)} bytes")
        return encoded_image.tobytes()
    else:
        raise ValueError("❌ Failed to encode image")


@app.websocket("/video_feed")
async def video_feed(websocket: WebSocket):

    await websocket.accept()

    # เปิดกล้อง (ใช้ index 0 สำหรับกล้องหลัก)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not found! Trying to reconnect...")
        time.sleep(2)
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        # ส่งข้อความไปที่ frontend ถ้ากล้องไม่พร้อม
        await websocket.send_text("Unable to connect camera, please check the camera.")
        cap.release()
        return

    image_count = 0  # นับจำนวนภาพที่บันทึก
    last_detection_time = 0
    #time delay for crop images
    detection_delay = 2 

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
            if len(faces) > 0:
                current_time = time.time()  # เวลาปัจจุบันในวินาที

                # ตรวจสอบว่าเวลาแตกต่างจากการตรวจพบครั้งล่าสุดเกิน 2 วินาทีหรือไม่
                if current_time - last_detection_time >= detection_delay:
                    # เริ่ม crop ภาพใบหน้าเมื่อเวลาผ่านไป 2 วินาที
                    last_detection_time = current_time  # อัพเดทเวลาการตรวจพบล่าสุด
                    
                    padding = 100

                    # Crop และบันทึกใบหน้า
                    for (x, y, w, h) in faces:
                        # หาขนาดที่ใหญ่ที่สุด (กว้างหรือสูง)
                        max_size = max(w, h)

                        # คำนวณพิกัดใหม่ให้ขยายออกแบบสมมาตร
                        x1 = max(x + w//2 - max_size//2 - padding, 0)  
                        x2 = min(x + w//2 + max_size//2 + padding, frame.shape[1])  
                        y1 = max(y + h//2 - max_size//2 - padding, 0)  
                        y2 = min(y + h//2 + max_size//2 + padding, frame.shape[0])  

                        # Crop ภาพใบหน้าที่ถูกขยาย
                        face_crop = frame[y1:y2, x1:x2]

                        # รีไซส์ภาพให้มีขนาดคงที่ (เช่น 160x160)
                        face_crop_resized = cv2.resize(face_crop, (160, 160))

                        # บันทึกภาพ
                        image_count += 1
                        filename = os.path.join(output_folder, f"face_{image_count}.jpg")
                        cv2.imwrite(filename, face_crop_resized)
                        print(f"บันทึกใบหน้าที่ {image_count}: {filename}")

                        image_bytes = compress_and_encode_image(face_crop, quality=100)
                        send_image_to_rabbitmq(channel, image_bytes, camera_id)

            else:
                # ไม่มีการตรวจพบใบหน้า
                last_detection_time = 0  # รีเซ็ตเวลาถ้าคุณไม่ต้องการทำอะไรเมื่อไม่มีใบหน้า

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