from fastapi import FastAPI, WebSocket
from app.rabbitmq import *

#import lib
import cv2
import numpy as np
import base64
import asyncio
import time
import os


app = FastAPI()

# Haar Cascade model 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Connection RabbitMQ always
connection, channel = get_rabbitmq_connection()

output_folder = "faces"
os.makedirs(output_folder, exist_ok=True)  # สร้างโฟลเดอร์ถ้ายังไม่มี

# Camera id and can implement 
camera_id = "CAM_01"

def compress_and_encode_image(image, quality=100):
    """ Compress the image and encode it into bytes. """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be numpy array")

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    success, encoded_image = cv2.imencode(".jpg", image, encode_param)
    
    if success:
        print(f"✅ Encoded image size: {len(encoded_image)} bytes")
        return encoded_image.tobytes()
    else:
        raise ValueError("❌ Failed to encode image")


@app.websocket("/video_feed")
async def video_feed(websocket: WebSocket):

    await websocket.accept()

    # Open camera (Use index 0 for main camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not found! Trying to reconnect...")
        time.sleep(2)
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        # Send Message to frontend if camera not ready
        await websocket.send_text("Unable to connect camera, please check the camera.")
        cap.release()
        return

    image_count = 0  # นับจำนวนภาพที่บันทึก
    last_detection_time = 0
    #time delay for crop images
    detection_delay = 4 

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await websocket.send_text("The camera is not turned on.")
                break

            # Convert image to Grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Face detection
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                current_time = time.time()  # Current time 

                # Check if the time is more than 4 seconds different from the last detection.
                if current_time - last_detection_time >= detection_delay:
                    # Start cropping the face image after 4 seconds.
                    last_detection_time = current_time  # อัพเดทเวลาการตรวจพบล่าสุด
                    
                    padding = 100

                    # Crop and record faces
                    for (x, y, w, h) in faces:
                        max_size = max(w, h)

                        # Size for crop face image
                        x1 = max(x + w//2 - max_size//2 - padding, 0)  
                        x2 = min(x + w//2 + max_size//2 + padding, frame.shape[1])  
                        y1 = max(y + h//2 - max_size//2 - padding, 0)  
                        y2 = min(y + h//2 + max_size//2 + padding, frame.shape[0])  

                        # Crop iamge
                        face_crop = frame[y1:y2, x1:x2]

                        # Resize the image to a fixed size (เช่น 160x160)
                        face_crop_resized = cv2.resize(face_crop, (160, 160))

                        # record images to faces floder
                        image_count += 1
                        filename = os.path.join(output_folder, f"face_{image_count}.jpg")
                        cv2.imwrite(filename, face_crop_resized)
                        print(f"record {image_count}: {filename}")

                        # send image to Compress the image and encode it into bytes
                        image_bytes = compress_and_encode_image(face_crop, quality=100)

                        # Send image_bytes to RabbitMQ
                        send_image_to_rabbitmq(channel, image_bytes, camera_id)

            else:
                # No face detected
                last_detection_time = 0  # Reset time 

            # Draw a frame around the face.
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Convert images to base64 for sending via WebSocket
            _, buffer = cv2.imencode(".jpg", frame)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")

            # send image to Client
            await websocket.send_text(frame_base64)

    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        cap.release()
        await websocket.close()