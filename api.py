import json
import logging
import os
import threading
from typing import Optional, List, Dict, Any, Tuple

import pika
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from face_service.core.recognize import recognize_b64
from logic.dataset_store import make_redis, save_dataset, delete_dataset, load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("face-service")

app = FastAPI()
rdb = make_redis()

# -----------------------------
# Dataset API
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
# Consumer state
# -----------------------------
class ConnectIn(BaseModel):
    lecture_id: str = Field(..., min_length=1)
    in_amqp_url: str = Field(..., min_length=1)
    in_queue: str = Field(..., min_length=1)
    threshold: float = 0.45

_worker_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()

_last_result_lock = threading.Lock()
_last_result: Optional[Dict[str, Any]] = None

_state_lock = threading.Lock()
_state: Dict[str, Any] = {
    "running": False,
    "lecture_id": None,
    "in_queue": None,
    "in_amqp_url": None,
    "out_queue": None,
    "out_amqp_url": None,
    "last_error": None,
}

# -----------------------------
# OUT resolve from ENV
# -----------------------------
def resolve_out_from_env() -> Tuple[str, str]:
    out_amqp_url = os.getenv("OUT_AMQP_URL")
    out_queue = os.getenv("OUT_QUEUE")

    if not out_amqp_url or not out_queue:
        raise ValueError("OUT is not configured: set OUT_AMQP_URL and OUT_QUEUE env vars")

    return out_amqp_url, out_queue

def _test_rabbit_connect(amqp_url: str) -> None:
    params = pika.URLParameters(amqp_url)
    params.socket_timeout = 5
    params.connection_attempts = 1
    params.retry_delay = 0
    conn = pika.BlockingConnection(params)
    conn.close()

def start_consumer(cfg: ConnectIn):
    global _last_result

    # -----------------------------
    # prepare OUT connection once
    # -----------------------------
    try:
        out_amqp_url, out_queue = resolve_out_from_env()
    except Exception as e:
        with _state_lock:
            _state["last_error"] = str(e)
            _state["running"] = False
        log.error("[consumer] OUT env missing: %s", e)
        return

    with _state_lock:
        _state.update(
            running=True,
            lecture_id=cfg.lecture_id,
            in_queue=cfg.in_queue,
            in_amqp_url=cfg.in_amqp_url,
            out_queue=out_queue,
            out_amqp_url=out_amqp_url,
            last_error=None,
        )

    try:
        # IN connection (Дима -> я)
        in_params = pika.URLParameters(cfg.in_amqp_url)
        in_params.heartbeat = 30
        in_params.blocked_connection_timeout = 30
        in_conn = pika.BlockingConnection(in_params)
        in_ch = in_conn.channel()
        in_ch.queue_declare(queue=cfg.in_queue, durable=True)
        in_ch.basic_qos(prefetch_count=1)

        # OUT connection (я -> rabbit -> Артём читает)
        out_params = pika.URLParameters(out_amqp_url)
        out_params.heartbeat = 30
        out_params.blocked_connection_timeout = 30
        out_conn = pika.BlockingConnection(out_params)
        out_ch = out_conn.channel()
        out_ch.queue_declare(queue=out_queue, durable=True)

        log.info(
            "[consumer] started lecture_id=%s in_queue=%s out_queue=%s",
            cfg.lecture_id,
            cfg.in_queue,
            out_queue,
        )

        def publish_out(payload: Dict[str, Any]) -> None:
            out_ch.basic_publish(
                exchange="",
                routing_key=out_queue,  # IMPORTANT: resolved out_queue
                body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            )

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
                            "error": None,
                        }

                with _last_result_lock:
                    _last_result = result

                # -----------------------------
                # ADDED: publish to OUT for Artem
                # -----------------------------
                publish_out(result)

                log.info("[consumer] result %s", result)

            except Exception as e:
                err = {
                    "lecture_id": cfg.lecture_id,
                    "request_id": None,
                    "person_id": None,
                    "error": str(e),
                }
                with _last_result_lock:
                    _last_result = err

                # publish error тоже (чтобы Артём видел)
                try:
                    publish_out(err)
                except Exception:
                    log.exception("[consumer] failed to publish error to OUT")

                log.exception("[consumer] error while processing message")

            finally:
                ch.basic_ack(delivery_tag=method.delivery_tag)

        in_ch.basic_consume(queue=cfg.in_queue, on_message_callback=on_message)

        _stop_event.clear()
        while not _stop_event.is_set():
            in_conn.process_data_events(time_limit=1.0)

        # graceful close
        try:
            in_conn.close()
        except Exception:
            pass
        try:
            out_conn.close()
        except Exception:
            pass

        log.info("[consumer] stopped")

    except Exception as e:
        with _state_lock:
            _state["last_error"] = str(e)
        log.exception("[consumer] crashed")

    finally:
        with _state_lock:
            _state["running"] = False

@app.post("/connect")
def connect(cfg: ConnectIn):
    global _worker_thread

    with _state_lock:
        if _state["running"]:
            return {"ok": False, "error": "already connected"}

    # ADDED: validate OUT env early, so /connect fails fast if missing
    try:
        out_amqp_url, out_queue = resolve_out_from_env()
        _test_rabbit_connect(out_amqp_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"cannot connect to OUT: {e}")

    # 1) test IN
    try:
        _test_rabbit_connect(cfg.in_amqp_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"cannot connect to in_amqp_url: {e}")

    _worker_thread = threading.Thread(target=start_consumer, args=(cfg,), daemon=True)
    _worker_thread.start()
    return {
        "ok": True,
        "lecture_id": cfg.lecture_id,
        "status": "connected",
        "out_queue": out_queue,
    }

@app.post("/disconnect")
def disconnect():
    _stop_event.set()
    return {"ok": True, "status": "stopping"}

@app.get("/status")
def status():
    with _state_lock:
        return {"ok": True, **_state}

@app.get("/last_result")
def last_result():
    with _last_result_lock:
        return {"ok": True, "result": _last_result}

@app.get("/health")
def health():
    return {"ok": True}
