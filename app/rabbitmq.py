import threading
import base64
import pika
import logging
import time
import json
import zlib

# setting log
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RABBITMQ_HOST = "SY_rabbitmq"
RABBITMQ_USER = "S@ony_devide0102"
RABBITMQ_PASS = "S@ony_devide0102"
QUEUE_NAME = "face_images"

# RabbitMQ connrttion function
def get_rabbitmq_connection():
    try:
        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        connection = pika.BlockingConnection(pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            credentials=credentials,
            heartbeat=120  # timeout protaction
        ))
        channel = connection.channel()
        channel.queue_declare(queue=QUEUE_NAME, durable=True)
        logging.info("✅ RabbitMQ connection established successfully")
        return connection, channel
    except Exception as e:
        logging.error(f"❌ Failed to connect to RabbitMQ: {type(e).__name__}: {e}")
        raise


# Send image to RabbitMQ
def send_image_to_rabbitmq(channel, image_bytes, camera_id):
    """ Use Thread to send images to RabbitMQ to reduce WebSocket latency. """
    thread = threading.Thread(target=send_to_rabbitmq, args=(channel, image_bytes, camera_id))
    thread.start()

# ฟังก์ชันส่งภาพไปยัง RabbitMQ
def send_to_rabbitmq(channel, image_bytes, camera_id):
    """ Send images to RabbitMQ with Camera ID """
    try:
        print(f"📌 Debug: Size image_bytes ก่อนบีบอัด: {len(image_bytes)} bytes")
        
        # บีบอัดข้อมูล (ลดขนาด Base64)
        compressed_data = zlib.compress(image_bytes, level=6)
        print(f"📌 Debug: ขนาดของ compressed_data หลังบีบอัด: {len(compressed_data)} bytes")

        image_base64 = base64.b64encode(compressed_data).decode("utf-8")
        print(f"📌 Debug: ขนาดของ image_base64 หลัง Base64 encode: {len(image_base64)} characters")

        message = {
            "camera_id": camera_id,
            "image": image_base64
        }

        # ตรวจสอบข้อความก่อนส่ง
        print(f"📩 Debug: Message ก่อนส่งเข้า RabbitMQ:\n{json.dumps(message)[:200]}...")  # แสดงแค่ 200 ตัวแรกเพื่อไม่ให้ log ยาวเกินไป

        # ส่งข้อมูลไปยัง RabbitMQ
        channel.basic_publish(
            exchange="",
            routing_key=QUEUE_NAME,
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Persistent Message
            )
        )

        print(f"✅ ส่งภาพจากกล้อง {camera_id} ไปยัง RabbitMQ สำเร็จ")

    except Exception as e:
        print(f"❌ ส่งภาพไม่สำเร็จ: {e}")

# เชื่อมต่อ RabbitMQ ก่อนเริ่มการทำงาน
connection, channel = get_rabbitmq_connection()
