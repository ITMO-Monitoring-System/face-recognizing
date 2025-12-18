import os, json, sys
import base64
import pika

def env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None or v == "":
        raise RuntimeError(f"Environment variable {name} is required")
    return v

TASK_QUEUE = env("TASK_QUEUE", "faces_tasks")

RABBIT_HOST = os.getenv("RABBIT_HOST", "localhost")
RABBIT_PORT = int(os.getenv("RABBIT_PORT", "5572"))
RABBIT_USER = os.getenv("RABBIT_USER", "user")
RABBIT_PASS = os.getenv("RABBIT_PASS", "pass")

def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "probe.jpg"
    request_id = sys.argv[2] if len(sys.argv) > 2 else "test-1"

    creds = pika.PlainCredentials(RABBIT_USER, RABBIT_PASS)
    params = pika.ConnectionParameters(host=RABBIT_HOST, port=RABBIT_PORT, credentials=creds)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.queue_declare(queue=TASK_QUEUE, durable=True)

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("ascii")

    msg = {"request_id": request_id, "image_b64": image_b64, "threshold": 0.45}

    ch.basic_publish(exchange="", routing_key=TASK_QUEUE, body=json.dumps(msg).encode("utf-8"))
    print("[send_task] sent:", msg)
    conn.close()

if __name__ == "__main__":
    main()
