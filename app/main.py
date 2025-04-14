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
camera_status = Gauge("camera_status", "Status of the camera (1=open, 0=closed)", ['camera_id'])
cpu_usage = Gauge("cpu_usage", "CPU usage in percentage")
ram_usage = Gauge("ram_usage", "RAM usage in percentage")

# RabbitMQ connection
connection, channel = get_rabbitmq_connection()


# Camera paths
camera_paths = {
    "CAM_01": "/dev/video0",
}

# Globals for each camera
latest_frames = {}
frame_locks = {}

# Initialize metrics
for cam_id in camera_paths:
    camera_status.labels(camera_id=cam_id).set(0)
cpu_usage.set(0)
ram_usage.set(0)

# ----------------- UTILS ------------------

def get_image_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def compress_and_encode_image(image, quality=100):
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be numpy array")
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded_image = cv2.imencode(".jpg", image, encode_param)
    if success:
        return encoded_image.tobytes()
    raise ValueError("Failed to encode image")

def resize_image(image, size=(112, 112)):
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

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
        cpu_usage.set(cpu_value)
        ram_usage.set(ram_value)
        await asyncio.sleep(5)

# ----------------- CAMERA WORKER ------------------

async def camera_worker(camera_id: str, camera_path: str):
    global latest_frames

    cap = cv2.VideoCapture(camera_path)
    if not cap.isOpened():
        logging.error(f"❌ Failed to open camera {camera_id} at {camera_path}")
        return

    logging.info(f"✅ Successfully opened camera {camera_id} at {camera_path}")
    camera_status.labels(camera_id=camera_id).set(1)

    last_detection_time = 0
    detection_delay = 3

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(1)
                continue

            frame = cv2.flip(frame, 1)
            results = yolo_model(frame)[0]
            current_time = time.time()

            if results.boxes:
                best_face = None
                best_score = 0

                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cropped_face = expand_crop(frame, x1, y1, x2, y2, expand_ratio=0.6)
                    resized_face = resize_image(cropped_face, size=(224, 224))
                    score = get_image_sharpness(resized_face)

                    if score > best_score:
                        best_score = score
                        best_face = resized_face

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if best_face is not None and best_score > 70 and current_time - last_detection_time >= detection_delay:
                    last_detection_time = current_time
                    image_bytes = compress_and_encode_image(best_face)

                    save_dir = f"app/saved_faces/{camera_id}"
                    os.makedirs(save_dir, exist_ok=True)
                    timestamp = int(time.time() * 1000)
                    save_path = os.path.join(save_dir, f"face_{timestamp}.jpg")
                    cv2.imwrite(save_path, best_face)

                    safe_publish(image_bytes, camera_id)

            with frame_locks[camera_id]:
                latest_frames[camera_id] = frame.copy()

            await asyncio.sleep(0.03)

    except Exception as e:
        logging.error(f"Camera Worker Error ({camera_id}): {e}")
    finally:
        cap.release()
        camera_status.labels(camera_id=camera_id).set(0)
        logging.info(f"🛑 Camera {camera_id} released.")

# ----------------- STARTUP EVENT ------------------

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(update_metrics())
    for cam_id, cam_path in camera_paths.items():
        frame_locks[cam_id] = Lock()
        latest_frames[cam_id] = None
        asyncio.create_task(camera_worker(cam_id, cam_path))

# ----------------- VIDEO STREAM (WebSocket) ------------------

@app.websocket("/video_feed/{camera_id}")
async def video_feed(websocket: WebSocket, camera_id: str):
    await websocket.accept()
    try:
        while True:
            if camera_id not in latest_frames:
                await websocket.send_text("Invalid camera ID")
                break

            with frame_locks[camera_id]:
                frame = latest_frames[camera_id].copy() if latest_frames[camera_id] is not None else None

            if websocket.client_state != WebSocketState.CONNECTED:
                break

            if frame is not None:
                _, buffer = cv2.imencode(".jpg", frame)
                frame_base64 = base64.b64encode(buffer).decode("utf-8")
                await websocket.send_text(frame_base64)
            else:
                await websocket.send_text("No frame available.")

            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        logging.info(f"WebSocket disconnected for {camera_id}")
    except Exception as e:
        logging.error(f"WebSocket Error ({camera_id}): {e}")
    finally:
        if websocket.application_state == WebSocketState.CONNECTED:
            try:
                await websocket.close()
            except RuntimeError:
                pass
