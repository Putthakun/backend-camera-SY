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


# Split thread for send images to RabbitMQ
def send_image_to_rabbitmq(channel, image_bytes, camera_id):
    """ Use Thread to send images to RabbitMQ to reduce WebSocket latency. """
    thread = threading.Thread(target=send_to_rabbitmq, args=(channel, image_bytes, camera_id))
    thread.start()

# Send images to RabbitMQ
def send_to_rabbitmq(channel, image_bytes, camera_id):
    """ Send images to RabbitMQ with Camera ID """
    try:
        # Compress data (Reduce Base64)
        compressed_data = zlib.compress(image_bytes, level=6)

        image_base64 = base64.b64encode(compressed_data).decode("utf-8")

        # Message send to RabbitMQ
        message = {
            "camera_id": camera_id,
            "image": image_base64
        }

        # Send data to RabbitMQ
        channel.basic_publish(
            exchange="",
            routing_key=QUEUE_NAME,   # QUEUE_NAME = face_images
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Persistent Message
            )
        )

        print(f"✅ Send image from {camera_id} to RabbitMQ successfully")

    except Exception as e:
        print(f"❌ Failed to send image: {e}")

# Connect RabbitMQ before starting work
connection, channel = get_rabbitmq_connection()
