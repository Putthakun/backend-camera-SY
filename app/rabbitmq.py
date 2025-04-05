import threading
import base64
import pika
import logging
import time
import json
import zlib

# ตั้งค่าการ log
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# RabbitMQ config
RABBITMQ_HOST = "SY_rabbitmq"
RABBITMQ_USER = "S@ony_devide0102"
RABBITMQ_PASS = "S@ony_devide0102"
QUEUE_NAME = "face_images"

# ตัวแปร global สำหรับเชื่อมต่อ RabbitMQ
connection = None
channel = None


def get_rabbitmq_connection(retries=5, delay=5):
    """ พยายามเชื่อมต่อ RabbitMQ ใหม่เมื่อเชื่อมต่อไม่สำเร็จ """
    attempt = 0
    while attempt < retries:
        try:
            credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
            parameters = pika.ConnectionParameters(
                host=RABBITMQ_HOST,
                credentials=credentials,
                heartbeat=120,
                blocked_connection_timeout=30
            )
            conn = pika.BlockingConnection(parameters)
            ch = conn.channel()
            ch.queue_declare(queue=QUEUE_NAME, durable=True)
            logging.info("✅ RabbitMQ connection established successfully")
            return conn, ch
        except pika.exceptions.AMQPConnectionError as e:
            attempt += 1
            logging.warning(f"❌ RabbitMQ connection failed: {e}. Retrying in {delay}s... ({attempt}/{retries})")
            time.sleep(delay)
        except Exception as e:
            logging.error(f"❌ Unexpected error: {type(e).__name__}: {e}")
            break
    raise ConnectionError("❌ Failed to connect to RabbitMQ after retries.")


def safe_publish(image_bytes, camera_id):
    """ ตรวจสอบสถานะก่อนส่ง หาก connection/channel ปิด จะ reconnect ใหม่ """
    global connection, channel

    if connection is None or connection.is_closed or channel is None or channel.is_closed:
        logging.warning("🔄 RabbitMQ connection or channel is closed. Reconnecting...")
        try:
            connection, channel = get_rabbitmq_connection()
        except Exception as e:
            logging.error(f"❌ Cannot reconnect to RabbitMQ: {e}")
            return

    # ✅ ส่งผ่าน thread
    thread = threading.Thread(target=send_to_rabbitmq, args=(channel, image_bytes, camera_id))
    thread.start()


def send_to_rabbitmq(ch, image_bytes, camera_id):
    """ ส่งภาพไปยัง RabbitMQ พร้อม camera_id """
    try:
        compressed_data = zlib.compress(image_bytes, level=6)
        image_base64 = base64.b64encode(compressed_data).decode("utf-8")

        message = {
            "camera_id": camera_id,
            "image": image_base64
        }

        ch.basic_publish(
            exchange="",
            routing_key=QUEUE_NAME,
            body=json.dumps(message),
            properties=pika.BasicProperties(delivery_mode=2)
        )

        logging.info(f"✅ Sent image from {camera_id} to RabbitMQ")
    except Exception as e:
        logging.error(f"❌ Failed to send image: {e}")


# ✅ เรียกครั้งแรกเพื่อเปิด connection
connection, channel = get_rabbitmq_connection()

# 🔁 ตัวอย่างการใช้งาน (คุณจะเรียก safe_publish(...) แทน)
# safe_publish(image_bytes, "CAM_01")
