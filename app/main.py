from fastapi import FastAPI, WebSocket, Response
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
import mediapipe as mp
from threading import Lock

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# BlazeFace (MediaPipe) setup
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

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

def crop_face_with_margin_px(frame, bbox, top=150, bottom=150, left=150, right=150):
    img_h, img_w, _ = frame.shape
    x = int(bbox.xmin * img_w)
    y = int(bbox.ymin * img_h)
    w = int(bbox.width * img_w)
    h = int(bbox.height * img_h)
    x1 = max(x - left, 0)
    y1 = max(y - top, 0)
    x2 = min(x + w + right, img_w)
    y2 = min(y + h + bottom, img_h)
    return frame[y1:y2, x1:x2]

def resize_image(image, size=(224, 224)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("❌ Failed to open camera in background task.")
        return

    camera_status.set(1)
    last_detection_time = 0
    detection_delay = 3

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("⚠️ Failed to read frame.")
                await asyncio.sleep(1)
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb_frame)

            current_time = time.time()
            if results.detections and (current_time - last_detection_time >= detection_delay):
                last_detection_time = current_time
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    cropped_face = crop_face_with_margin_px(frame, bbox)
                    resized_face = resize_image(cropped_face, size=(224, 224))
                    image_bytes = compress_and_encode_image(resized_face)
                    safe_publish(image_bytes, camera_id)

            if results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x = int(bbox.xmin * iw)
                    y = int(bbox.ymin * ih)
                    w = int(bbox.width * iw)
                    h = int(bbox.height * ih)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            with frame_lock:
                latest_frame = frame.copy()

            await asyncio.sleep(0.03)

    except Exception as e:
        logging.error(f"📸 Camera Worker Error: {e}")
    finally:
        cap.release()
        camera_status.set(0)

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
            if frame is not None:
                _, buffer = cv2.imencode(".jpg", frame)
                frame_base64 = base64.b64encode(buffer).decode("utf-8")
                await websocket.send_text(frame_base64)
            else:
                await websocket.send_text("No frame available.")
            await asyncio.sleep(0.03)
    except Exception as e:
        logging.error(f"WebSocket Error: {e}")
    finally:
        await websocket.close()
