import json
import threading
from typing import Optional, List, Dict, Any

import pika
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from face_service.core.recognize import recognize_b64
from logic.dataset_store import make_redis, save_dataset, delete_dataset, load_dataset

app = FastAPI()
rdb = make_redis()

# -----------------------------
# Dataset API (оставляем как есть)
# -----------------------------
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


# -----------------------------
# Consumer (ТОЛЬКО IN, без OUT)
# -----------------------------
_worker_thread: Optional[threading.Thread] = None
_last_result_lock = threading.Lock()
_last_result: Optional[Dict[str, Any]] = None

class ConnectIn(BaseModel):
    lecture_id: str = Field(..., min_length=1)
    in_amqp_url: str = Field(..., min_length=1)
    in_queue: str = Field(..., min_length=1)
    threshold: float = 0.45

def start_consumer(cfg: ConnectIn):
    global _last_result

    in_conn = pika.BlockingConnection(pika.URLParameters(cfg.in_amqp_url))
    in_ch = in_conn.channel()
    in_ch.queue_declare(queue=cfg.in_queue, durable=True)
    in_ch.basic_qos(prefetch_count=1)

    print(f"[consumer] started: lecture_id={cfg.lecture_id}, in_queue={cfg.in_queue}")

    def on_message(ch, method, props, body: bytes):
        global _last_result

        try:
            msg = json.loads(body.decode("utf-8"))
            request_id = msg.get("request_id")
            image_b64 = msg.get("image_b64")

            if not request_id or not image_b64:
                result = {
                    "lecture_id": cfg.lecture_id,
                    "request_id": request_id,
                    "person_id": None,
                    "error": "bad_message",
                }
            else:
                persons = load_dataset(rdb, cfg.lecture_id)
                if not persons:
                    result = {
                        "lecture_id": cfg.lecture_id,
                        "request_id": request_id,
                        "person_id": None,
                        "error": "dataset_not_loaded",
                    }
                else:
                    faces = recognize_b64(
                        image_b64,
                        persons,
                        threshold=float(msg.get("threshold", cfg.threshold)),
                    )

                    person_id = None
                    for f in faces:
                        if f.get("matched"):
                            person_id = f.get("person_id")
                            break

                    result = {
                        "lecture_id": cfg.lecture_id,
                        "request_id": request_id,
                        "person_id": person_id,
                    }

            # Сохраним последний результат для дебага
            with _last_result_lock:
                _last_result = result

            print(f"[consumer] result: {result}")

        except Exception as e:
            err = {
                "lecture_id": cfg.lecture_id,
                "request_id": None,
                "person_id": None,
                "error": str(e),
            }
            with _last_result_lock:
                _last_result = err
            print(f"[consumer] error: {err}")

        finally:
            # Для теста с Димой достаточно ack, чтобы очередь разгребалась
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

@app.get("/last_result")
def last_result():
    with _last_result_lock:
        return {"ok": True, "result": _last_result}

@app.get("/health")
def health():
    return {"ok": True}
