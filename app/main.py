from fastapi import FastAPI, WebSocket, Response
from starlette.websockets import WebSocketDisconnect, WebSocketState
from fastapi.responses import PlainTextResponse
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST

# custom modules
from app.rabbitmq import get_rabbitmq_connection, safe_publish

# libraries
import cv2
import numpy as np
import base64
import asyncio
import time
import os
import psutil
import logging
from threading import Lock
from ultralytics import YOLO

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# YOLOv8-Face model
yolo_model = YOLO("app/yolov8n-face.pt")

# FastAPI app
app = FastAPI()

# Prometheus metrics
camera_status = Gauge("camera_status", "Status of the camera (1=open, 0=closed)")
cpu_usage = Gauge("cpu_usage", "CPU usage in percentage")
ram_usage = Gauge("ram_usage", "RAM usage in percentage")

# RabbitMQ connection
connection, channel = get_rabbitmq_connection()

# Camera ID
camera_id = "CAM_01"

# Initialize metrics
camera_status.set(0)
cpu_usage.set(0)
ram_usage.set(0)

# Shared global frame for WebSocket
latest_frame = None
frame_lock = Lock()

# ----------------- UTILS ------------------

def compress_and_encode_image(image, quality=100):
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be numpy array")
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded_image = cv2.imencode(".jpg", image, encode_param)
    if success:
        return encoded_image.tobytes()
    raise ValueError("Failed to encode image")

def resize_image(image, size=(224, 224)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def expand_crop(frame, x1, y1, x2, y2, expand_ratio=0.6):

    h, w, _ = frame.shape
    bw = x2 - x1
    bh = y2 - y1
    expand_w = int(bw * expand_ratio)
    expand_h = int(bh * expand_ratio)

    new_x1 = max(x1 - expand_w, 0)
    new_y1 = max(y1 - expand_h, 0)
    new_x2 = min(x2 + expand_w, w)
    new_y2 = min(y2 + expand_h, h)

    return frame[new_y1:new_y2, new_x1:new_x2]

# ----------------- METRICS ------------------

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

async def update_metrics():
    while True:
        cpu_value = psutil.cpu_percent()
        ram_value = psutil.virtual_memory().percent
        camera_value = camera_status.collect()[0].samples[0].value
        cpu_usage.set(cpu_value)
        ram_usage.set(ram_value)
        logging.info(f"Updated Metrics: CPU={cpu_value}%, RAM={ram_value}%, Camera={camera_value}")
        await asyncio.sleep(5)

# ----------------- CAMERA WORKER ------------------

async def camera_worker():
    global latest_frame

    camera_path = "/dev/video0"
    logging.info(f"📷 Trying to open camera at {camera_path} ...")

    cap = cv2.VideoCapture(camera_path)
    if not cap.isOpened():
        logging.error(f"❌ Failed to open camera at {camera_path}")
        return

    logging.info(f"✅ Successfully opened camera at {camera_path}")
    camera_status.set(1)

    last_detection_time = 0
    detection_delay = 3

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("⚠️ Failed to read frame from camera.")
                await asyncio.sleep(1)
                continue

            logging.debug("📸 Frame read successfully.")

            frame = cv2.flip(frame, 1)
            results = yolo_model(frame)[0]
            current_time = time.time()

            if results.boxes:
                logging.info(f"🧠 {len(results.boxes)} face(s) detected.")
                if current_time - last_detection_time >= detection_delay:
                    last_detection_time = current_time
                    for box in results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cropped_face = expand_crop(frame, x1, y1, x2, y2, expand_ratio=0.6)
                        resized_face = resize_image(cropped_face, size=(224, 224))
                        image_bytes = compress_and_encode_image(resized_face)

                        save_dir = "app/saved_faces"
                        os.makedirs(save_dir, exist_ok=True)
                        timestamp = int(time.time() * 1000)
                        save_path = os.path.join(save_dir, f"face_{timestamp}.jpg")
                        cv2.imwrite(save_path, resized_face)
                        logging.info(f"💾 Saved face image to {save_path}")

                        safe_publish(image_bytes, camera_id)
                        logging.info("📤 Published face to RabbitMQ.")

                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            with frame_lock:
                latest_frame = frame.copy()

            await asyncio.sleep(0.03)

    except Exception as e:
        logging.error(f"📸 Camera Worker Error: {e}")
    finally:
        cap.release()
        camera_status.set(0)
        logging.info("🛑 Camera released.")

# ----------------- STARTUP EVENT ------------------

@app.on_event("startup")
async def startup_event():
    print("🚀 Starting Prometheus metrics update task...")
    asyncio.create_task(update_metrics())
    print("📸 Starting camera worker...")
    asyncio.create_task(camera_worker())

# ----------------- VIDEO STREAM (WebSocket) ------------------

@app.websocket("/video_feed")
async def video_feed(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            with frame_lock:
                frame = latest_frame.copy() if latest_frame is not None else None

            if websocket.client_state != WebSocketState.CONNECTED:
                logging.warning("WebSocket not connected. Stopping stream.")
                break

            if frame is not None:
                _, buffer = cv2.imencode(".jpg", frame)
                frame_base64 = base64.b64encode(buffer).decode("utf-8")
                await websocket.send_text(frame_base64)
            else:
                await websocket.send_text("No frame available.")

            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        logging.info("WebSocket disconnected by client.")
    except Exception as e:
        logging.error(f"WebSocket Error: {e}")
    finally:
        if websocket.application_state == WebSocketState.CONNECTED:
            try:
                await websocket.close()
            except RuntimeError as e:
                logging.warning(f"WebSocket already closed: {e}")