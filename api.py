import json
import threading
from typing import Optional

import pika
from fastapi import FastAPI
from pydantic import BaseModel

from typing import List
import numpy as np
from pydantic import BaseModel, Field

from face_service.core.recognize import recognize_b64
from face_service.core.persons_loader import PersonsSource, load_persons
from logic.dataset_store import make_redis, save_dataset, delete_dataset, load_dataset


app = FastAPI()

rdb = make_redis()

class PersonIn(BaseModel):
    person_id: str = Field(..., min_length=1)
    embedding: List[float]

class DatasetIn(BaseModel):
    lecture_id: str = Field(..., min_length=1)
    persons: List[PersonIn]

@app.post("/dataset")
def set_dataset(payload: DatasetIn):
    persons = {}
    for p in payload.persons:
        emb = np.array(p.embedding, dtype=np.float32)
        persons.setdefault(p.person_id, []).append(emb)

    save_dataset(rdb, payload.lecture_id, persons)
    return {"ok": True, "lecture_id": payload.lecture_id, "persons_count": len(persons)}

@app.delete("/dataset/{lecture_id}")
def drop_dataset(lecture_id: str):
    deleted = delete_dataset(rdb, lecture_id)
    return {"ok": True, "lecture_id": lecture_id, "deleted": deleted}

_worker_thread: Optional[threading.Thread] = None


class ConnectIn(BaseModel):
    lecture_id: str
    in_amqp_url: str
    in_queue: str


def start_consumer(cfg: ConnectIn):
    in_conn = pika.BlockingConnection(pika.URLParameters(cfg.in_amqp_url))
    in_ch = in_conn.channel()
    in_ch.queue_declare(queue=cfg.in_queue, durable=True)
    in_ch.basic_qos(prefetch_count=1)

    out_conn = pika.BlockingConnection(pika.URLParameters(cfg.out_amqp_url))
    out_ch = out_conn.channel()
    out_ch.queue_declare(queue=cfg.out_queue, durable=True)

    def on_message(ch, method, props, body: bytes):
        try:
            msg = json.loads(body.decode("utf-8"))
            request_id = msg.get("request_id")
            image_b64 = msg.get("image_b64")

            if not request_id or not image_b64:
                out = {"lecture_id": cfg.lecture_id, "request_id": request_id, "error": "bad message", "faces": []}
            else:
                persons = load_dataset(rdb, cfg.lecture_id)
                if not persons:
                    out = {
                        "lecture_id": cfg.lecture_id,
                        "request_id": request_id,
                        "person_id": None,
                        "error": "dataset_not_loaded",
                    }
                else:
                    faces = recognize_b64(image_b64, persons, threshold=float(msg.get("threshold", cfg.threshold)))

                    person_id = None
                    for f in faces:
                        if f.get("matched"):
                            person_id = f.get("person_id")
                            break

                    out = {
                        "lecture_id": cfg.lecture_id,
                        "request_id": request_id,
                        "person_id": person_id,
                    }

            out_ch.basic_publish(
                exchange="",
                routing_key=cfg.out_queue,
                body=json.dumps(out, ensure_ascii=False).encode("utf-8"),
            )
        except Exception as e:
            out = {"lecture_id": cfg.lecture_id, "request_id": None, "error": str(e), "faces": []}
            out_ch.basic_publish(
                exchange="",
                routing_key=cfg.out_queue,
                body=json.dumps(out, ensure_ascii=False).encode("utf-8"),
            )
        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)

    in_ch.basic_consume(queue=cfg.in_queue, on_message_callback=on_message)
    in_ch.start_consuming()


@app.post("/connect")
def connect(cfg: ConnectIn):
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        return {"ok": False, "error": "already connected"}

    _worker_thread = threading.Thread(target=start_consumer, args=(cfg,), daemon=True)
    _worker_thread.start()
    return {"ok": True, "lecture_id": cfg.lecture_id, "status": "connected"}


@app.get("/health")
def health():
    return {"ok": True}
