import os
import pika

def env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None or v == "":
        raise RuntimeError(f"Environment variable {name} is required")
    return v

RESULT_QUEUE = env("RESULT_QUEUE", "faces_results")
RABBIT_HOST = env("RABBIT_HOST", "localhost")
RABBIT_PORT = int(os.getenv("RABBIT_PORT", "5572"))
RABBIT_USER = os.getenv("RABBIT_USER", "user")
RABBIT_PASS = os.getenv("RABBIT_PASS", "pass")

def main():
    creds = pika.PlainCredentials(RABBIT_USER, RABBIT_PASS)
    params = pika.ConnectionParameters(host=RABBIT_HOST, port=RABBIT_PORT, credentials=creds)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.queue_declare(queue=RESULT_QUEUE, durable=True)

    method, props, body = ch.basic_get(queue=RESULT_QUEUE, auto_ack=True)
    if not body:
        print("[read_results] no messages")
    else:
        print("[read_results] got:", body.decode("utf-8"))
    conn.close()

if __name__ == "__main__":
    main()
