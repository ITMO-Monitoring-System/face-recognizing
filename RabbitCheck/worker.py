import os
import json

import pika

from face_service.core.recognize import recognize_b64
from logic.dataset_store import make_redis, load_dataset

TASK_QUEUE = os.getenv("TASK_QUEUE", "faces_tasks")
RESULT_QUEUE = os.getenv("RESULT_QUEUE", "faces_results")

LECTURE_ID = os.getenv("LECTURE_ID", "123")

def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required")
    return value

RABBIT_HOST = require_env("RABBIT_HOST")
RABBIT_PORT = int(require_env("RABBIT_PORT"))
RABBIT_USER = require_env("RABBIT_USER")
RABBIT_PASS = require_env("RABBIT_PASS")

rdb = make_redis()

def main():
    creds = pika.PlainCredentials(RABBIT_USER, RABBIT_PASS)
    params = pika.ConnectionParameters(host=RABBIT_HOST, port=RABBIT_PORT, credentials=creds)

    conn = pika.BlockingConnection(params)
    ch = conn.channel()

    ch.queue_declare(queue=TASK_QUEUE, durable=True)
    ch.queue_declare(queue=RESULT_QUEUE, durable=True)
    ch.basic_qos(prefetch_count=1)

    def on_message(channel, method, properties, body: bytes):
        try:
            msg = json.loads(body.decode("utf-8"))
            request_id = msg.get("request_id")
            thr = float(msg.get("threshold", 0.45))
            image_b64 = msg.get("image_b64")

            persons = load_dataset(rdb, LECTURE_ID)  # <-- ВОТ ТУТ грузим по lecture_id

            if not persons:
                out = {
                    "lecture_id": LECTURE_ID,
                    "request_id": request_id,
                    "person_id": None,
                    "error": "dataset_not_loaded",
                }
            elif not image_b64:
                out = {"lecture_id": LECTURE_ID, "request_id": request_id, "person_id": None, "error": "image_b64 is required"}
            else:
                faces = recognize_b64(image_b64, persons, threshold=thr)

                person_id = None
                for f in faces:
                    if f.get("matched"):
                        person_id = f.get("person_id")
                        break

                out = {"lecture_id": LECTURE_ID, "request_id": request_id, "person_id": person_id}

            ch.basic_publish(
                exchange="",
                routing_key=RESULT_QUEUE,
                body=json.dumps(out, ensure_ascii=False).encode("utf-8"),
            )

        except Exception as e:
            err = {"lecture_id": LECTURE_ID, "request_id": None, "person_id": None, "error": str(e)}
            ch.basic_publish(exchange="", routing_key=RESULT_QUEUE, body=json.dumps(err, ensure_ascii=False).encode("utf-8"))
        finally:
            channel.basic_ack(delivery_tag=method.delivery_tag)

    ch.basic_consume(queue=TASK_QUEUE, on_message_callback=on_message)
    print(f"[worker] waiting tasks from {TASK_QUEUE} lecture_id={LECTURE_ID} ...")
    ch.start_consuming()

if __name__ == "__main__":
    main()
