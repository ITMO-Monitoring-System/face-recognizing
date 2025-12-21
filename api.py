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
    persons: Dict[str, List[np.ndarray]] = {}
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
# ADDED: keep handles for graceful stop
# -----------------------------
_in_conn_lock = threading.Lock()
_in_conn: Optional[pika.BlockingConnection] = None
_in_ch: Optional[pika.channel.Channel] = None

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

class OutCtx:
    conn: Optional[pika.BlockingConnection] = None
    ch: Optional[pika.channel.Channel] = None

out = OutCtx()

def _make_blocking_conn(amqp_url: str) -> pika.BlockingConnection:
    params = pika.URLParameters(amqp_url)
    params.heartbeat = 30
    params.blocked_connection_timeout = 30
    params.connection_attempts = 3
    params.retry_delay = 1
    return pika.BlockingConnection(params)

def start_consumer(cfg: ConnectIn):
    global _last_result, _in_conn, _in_ch

    # 1) Resolve OUT from ENV
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

    # 2) Build IN connection + channel
    try:
        in_conn = _make_blocking_conn(cfg.in_amqp_url)
        in_ch = in_conn.channel()
        in_ch.queue_declare(queue=cfg.in_queue, durable=True)
        in_ch.basic_qos(prefetch_count=1)

        with _in_conn_lock:
            _in_conn = in_conn
            _in_ch = in_ch

        log.info("[consumer] started lecture_id=%s in_queue=%s out_queue=%s", cfg.lecture_id, cfg.in_queue, out_queue)

    except Exception as e:
        with _state_lock:
            _state["last_error"] = f"IN connect failed: {e}"
            _state["running"] = False
        log.exception("[consumer] IN connect failed")
        return

    # 3) Prepare OUT connection factory (reconnect on demand)
    out_lock = threading.Lock()
    # out_conn: Optional[pika.BlockingConnection] = None
    # out_ch: Optional[pika.channel.Channel] = None

    def ensure_out_ready():
        if out.conn and out.ch and not out.conn.is_closed:
            return
        out.conn = _make_blocking_conn(out_amqp_url)
        out.ch = out.conn.channel()
        out.ch.queue_declare(queue=out_queue, durable=True)

    def publish_out(payload: Dict[str, Any]) -> bool:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        with out_lock:
            try:
                ensure_out_ready()
                assert out.ch is not None
                out.ch.basic_publish(exchange="", routing_key=out_queue, body=body)
                return True
            except Exception as e:
                log.warning("[consumer] OUT publish failed: %s", e)
                try:
                    if out.conn and not out.conn.is_closed:
                        out.conn.close()
                except Exception:
                    pass
                out.conn = None
                out.ch = None
                return False

    # 4) Consume callback
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

            # Publish to OUT (for Artem). If OUT is down, we don't crash IN consumer.
            if not publish_out(result):
                with _state_lock:
                    _state["last_error"] = "OUT publish failed (will retry on next message)"

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

            publish_out(err)  # best-effort
            log.exception("[consumer] error while processing message")

        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)

    in_ch.basic_consume(queue=cfg.in_queue, on_message_callback=on_message)

    # 5) Run consuming loop (stable)
    try:
        in_ch.start_consuming()
    except Exception as e:
        with _state_lock:
            _state["last_error"] = f"consumer stopped: {e}"
        log.exception("[consumer] stopped with error")
    finally:
        # cleanup
        try:
            with out_lock:
                if out.conn and not out.conn.is_closed:
                    out.conn.close()
        except Exception:
            pass

        try:
            with _in_conn_lock:
                if _in_conn and not _in_conn.is_closed:
                    _in_conn.close()
        except Exception:
            pass

        with _in_conn_lock:
            _in_conn = None
            _in_ch = None

        with _state_lock:
            _state["running"] = False

        log.info("[consumer] stopped")

@app.post("/connect")
def connect(cfg: ConnectIn):
    global _worker_thread

    with _state_lock:
        if _state["running"]:
            return {"ok": False, "error": "already connected"}

    # Validate OUT env + connectivity
    try:
        out_amqp_url, out_queue = resolve_out_from_env()
        _test_rabbit_connect(out_amqp_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"cannot connect to OUT: {e}")

    # Validate IN connectivity
    try:
        _test_rabbit_connect(cfg.in_amqp_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"cannot connect to in_amqp_url: {e}")

    _worker_thread = threading.Thread(target=start_consumer, args=(cfg,), daemon=True)
    _worker_thread.start()
    return {"ok": True, "lecture_id": cfg.lecture_id, "status": "connected", "out_queue": out_queue}

@app.post("/disconnect")
def disconnect():
    # Graceful stop: request pika's ioloop to stop consuming
    with _in_conn_lock:
        conn = _in_conn
        ch = _in_ch

    if conn and ch and (not conn.is_closed) and (not ch.is_closed):
        try:
            conn.add_callback_threadsafe(ch.stop_consuming)
            return {"ok": True, "status": "stopping"}
        except Exception as e:
            with _state_lock:
                _state["last_error"] = f"disconnect failed: {e}"
            return {"ok": False, "error": str(e)}

    return {"ok": True, "status": "not running"}

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
