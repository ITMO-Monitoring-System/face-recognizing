import json
import logging
import os
import threading
from typing import Optional, List, Dict, Any, Tuple

import pika
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from face_service.core.recognize import recognize_b64
from face_service.core.recognize import best_embedding_bytes
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

class EmbeddingOut(BaseModel):
    ok: bool
    embedding: List[float]
    bbox: Optional[List[float]] = None


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

@app.post("/api/embedding", response_model=EmbeddingOut)
async def embedding_from_bytes(request: Request):
    img_bytes = await request.body()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="empty body")

    face = best_embedding_bytes(img_bytes)
    if face is None:
        raise HTTPException(status_code=404, detail="no face detected")

    emb = face["embedding"].astype(np.float32).tolist()
    bbox = [float(x) for x in face["bbox"]]

    # emb уже list[float] (через tolist), bbox тоже приводим к float
    return {"ok": True, "embedding": emb, "bbox": bbox}


class OutCtx:
    def __init__(self) -> None:
        self.conn: Optional[pika.BlockingConnection] = None
        self.ch: Optional[pika.channel.Channel] = None

    def close(self) -> None:
        try:
            if self.conn and not self.conn.is_closed:
                self.conn.close()
        except Exception:
            pass
        self.conn = None
        self.ch = None

# -----------------------------
# Lecture control API (multi-queue)
# -----------------------------
class LectureStartIn(BaseModel):
    lecture_id: str = Field(..., min_length=1)
    in_amqp_url: str = Field(..., min_length=1)
    in_queue: str = Field(..., min_length=1)
    threshold: float = 0.45


class LectureStopIn(BaseModel):
    lecture_id: str = Field(..., min_length=1)


# -----------------------------
# Runtime state
# -----------------------------
_lectures_lock = threading.Lock()


class LectureRuntime:
    def __init__(
        self,
        lecture_id: str,
        in_amqp_url: str,
        in_queue: str,
        out_amqp_url: str,
        out_queue: str,
        threshold: float,
    ):
        self.lecture_id = lecture_id
        self.in_amqp_url = in_amqp_url
        self.in_queue = in_queue
        self.out_amqp_url = out_amqp_url
        self.out_queue = out_queue
        self.threshold = threshold

        self.thread: Optional[threading.Thread] = None
        self.in_conn: Optional[pika.BlockingConnection] = None
        self.in_ch: Optional[pika.channel.Channel] = None

        self.running: bool = False
        self.last_error: Optional[str] = None


_lectures: Dict[str, LectureRuntime] = {}

_last_results_lock = threading.Lock()
_last_results: Dict[str, Dict[str, Any]] = {}  # lecture_id -> last result


# -----------------------------
# RabbitMQ helpers
# -----------------------------
def _make_blocking_conn(amqp_url: str) -> pika.BlockingConnection:
    params = pika.URLParameters(amqp_url)
    params.heartbeat = 30
    params.blocked_connection_timeout = 30
    params.connection_attempts = 3
    params.retry_delay = 1
    return pika.BlockingConnection(params)


def _test_rabbit_connect(amqp_url: str) -> None:
    params = pika.URLParameters(amqp_url)
    params.socket_timeout = 5
    params.connection_attempts = 1
    params.retry_delay = 0
    conn = pika.BlockingConnection(params)
    conn.close()


# -----------------------------
# OUT / Backend (Artem) helpers
# -----------------------------
def resolve_out_amqp_url_from_env() -> str:
    out_amqp_url = os.getenv("OUT_AMQP_URL")
    if not out_amqp_url:
        raise ValueError("OUT_AMQP_URL is not configured")
    return out_amqp_url


def resolve_out_queue_for_lecture(lecture_id: str) -> str:
    prefix = os.getenv("OUT_QUEUE_PREFIX")
    if not prefix:
        raise ValueError("OUT_QUEUE_PREFIX is not configured")
    return f"{prefix}.{lecture_id}"


def backend_cfg() -> Tuple[Optional[str], str, str]:
    base = os.getenv("BACKEND_URL")
    start_path = os.getenv("BACKEND_START_PATH", "/api/lecture/start")
    stop_path = os.getenv("BACKEND_STOP_PATH", "/api/lecture/stop")
    return base, start_path, stop_path


def _lecture_id_payload_value(lecture_id: str) -> Any:
    # У Артёма в swagger пример lecture_id как число.
    # Если у вас lecture_id всегда числовой — уйдет int.
    # Если нет — уйдет строка (не сломает ваш сервис, если он типизирует как string).
    return int(lecture_id) if lecture_id.isdigit() else lecture_id


def notify_backend_start(lecture_id: str, out_queue: str) -> None:
    base, start_path, _ = backend_cfg()
    if not base:
        return
    try:
        requests.post(
            f"{base}{start_path}",
            json={"lecture_id": _lecture_id_payload_value(lecture_id), "queue": out_queue},
            timeout=5,
        )
    except Exception as e:
        log.warning("[lecture=%s] backend start notify failed: %s", lecture_id, e)


def notify_backend_stop(lecture_id: str, out_queue: str) -> None:
    base, _, stop_path = backend_cfg()
    if not base:
        return
    try:
        requests.post(
            f"{base}{stop_path}",
            json={"lecture_id": _lecture_id_payload_value(lecture_id), "queue": out_queue},
            timeout=5,
        )
    except Exception as e:
        log.warning("[lecture=%s] backend stop notify failed: %s", lecture_id, e)


# -----------------------------
# Consumer loop per lecture
# -----------------------------
def run_lecture_consumer(rt: LectureRuntime) -> None:
    rt.running = True

    out_lock = threading.Lock()
    out_ctx = OutCtx()

    def ensure_out_ready() -> None:
        if out_ctx.conn and out_ctx.ch and not out_ctx.conn.is_closed:
            return

        out_ctx.conn = _make_blocking_conn(rt.out_amqp_url)
        out_ctx.ch = out_ctx.conn.channel()
        out_ctx.ch.queue_declare(queue=rt.out_queue, durable=True)

    def publish_out(payload: Dict[str, Any]) -> bool:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        with out_lock:
            try:
                ensure_out_ready()
                assert out_ctx.ch is not None
                out_ctx.ch.basic_publish(exchange="", routing_key=rt.out_queue, body=body)
                return True
            except Exception as e:
                rt.last_error = f"OUT publish failed: {e}"
                log.warning("[lecture=%s] OUT publish failed: %s", rt.lecture_id, e)
                out_ctx.close()
                return False

    try:
        # IN connect (RabbitMQ Димы)
        rt.in_conn = _make_blocking_conn(rt.in_amqp_url)
        rt.in_ch = rt.in_conn.channel()
        rt.in_ch.queue_declare(queue=rt.in_queue, durable=True)
        rt.in_ch.basic_qos(prefetch_count=1)

        log.info(
            "[lecture=%s] consumer started in_queue=%s out_queue=%s",
            rt.lecture_id,
            rt.in_queue,
            rt.out_queue,
        )

        # notify Artem: где читать результаты этой лекции
        notify_backend_start(rt.lecture_id, rt.out_queue)

        def on_message(ch, method, props, body: bytes):
            lecture_id = rt.lecture_id
            try:
                msg = json.loads(body.decode("utf-8"))
                image_b64 = msg.get("image_b64")
                thr = float(msg.get("threshold", rt.threshold))

                person_id = None

                if image_b64:
                    persons = load_dataset(rdb, lecture_id)
                    if persons:
                        faces = recognize_b64(image_b64, persons, threshold=thr)
                        for f in faces:
                            if f.get("matched"):
                                person_id = f.get("person_id")
                                break

                result = {
                    "lecture_id": lecture_id,
                    "person_id": person_id,
                }

                with _last_results_lock:
                    _last_results[lecture_id] = result

                if not publish_out(result):
                    rt.last_error = "OUT publish failed (will retry on next message)"

                log.info("[lecture=%s] result %s", lecture_id, result)

            except Exception as e:
                err = {
                    "lecture_id": lecture_id,
                    "person_id": None,
                }
                with _last_results_lock:
                    _last_results[lecture_id] = err
                publish_out(err)  # best-effort
                rt.last_error = str(e)
                log.exception("[lecture=%s] error while processing message", lecture_id)

            finally:
                ch.basic_ack(delivery_tag=method.delivery_tag)

        rt.in_ch.basic_consume(queue=rt.in_queue, on_message_callback=on_message)

        # consuming loop
        rt.in_ch.start_consuming()

    except Exception as e:
        rt.last_error = str(e)
        log.exception("[lecture=%s] consumer stopped with error", rt.lecture_id)

    finally:
        # notify Artem stop
        notify_backend_stop(rt.lecture_id, rt.out_queue)

        try:
            with out_lock:
                out_ctx.close()
        except Exception:
            pass

        try:
            if rt.in_conn and not rt.in_conn.is_closed:
                rt.in_conn.close()
        except Exception:
            pass

        rt.in_conn = None
        rt.in_ch = None
        rt.running = False
        log.info("[lecture=%s] consumer stopped", rt.lecture_id)


# -----------------------------
# HTTP endpoints (start/stop/status)
# -----------------------------
@app.post("/api/lecture/start")
def lecture_start(payload: LectureStartIn):
    # OUT env + connectivity
    try:
        out_amqp_url = resolve_out_amqp_url_from_env()
        out_queue = resolve_out_queue_for_lecture(payload.lecture_id)
        _test_rabbit_connect(out_amqp_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"cannot connect to OUT: {e}")

    # IN connectivity (RabbitMQ Димы)
    try:
        _test_rabbit_connect(payload.in_amqp_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"cannot connect to in_amqp_url: {e}")

    with _lectures_lock:
        existing = _lectures.get(payload.lecture_id)
        if existing and existing.running:
            return {
                "ok": True,
                "status": "already_running",
                "lecture_id": payload.lecture_id,
                "in_queue": existing.in_queue,
                "out_queue": existing.out_queue,
            }

        rt = LectureRuntime(
            lecture_id=payload.lecture_id,
            in_amqp_url=payload.in_amqp_url,
            in_queue=payload.in_queue,
            out_amqp_url=out_amqp_url,
            out_queue=out_queue,
            threshold=payload.threshold,
        )
        t = threading.Thread(target=run_lecture_consumer, args=(rt,), daemon=True)
        rt.thread = t
        _lectures[payload.lecture_id] = rt
        t.start()

    return {"ok": True, "status": "started", "lecture_id": payload.lecture_id, "out_queue": out_queue}


@app.post("/api/lecture/stop")
def lecture_stop(payload: LectureStopIn):
    with _lectures_lock:
        rt = _lectures.get(payload.lecture_id)
        if not rt:
            return {"ok": True, "status": "not_found", "lecture_id": payload.lecture_id}

        conn = rt.in_conn
        ch = rt.in_ch
        out_queue = rt.out_queue

    if conn and ch and (not conn.is_closed) and (not ch.is_closed):
        try:
            conn.add_callback_threadsafe(ch.stop_consuming)
            return {
                "ok": True,
                "status": "stopping",
                "lecture_id": payload.lecture_id,
                "out_queue": out_queue,
            }
        except Exception as e:
            rt.last_error = str(e)
            return {"ok": False, "error": str(e), "lecture_id": payload.lecture_id}

    return {"ok": True, "status": "not_running", "lecture_id": payload.lecture_id, "out_queue": out_queue}


@app.get("/api/lecture/status")
def lecture_status():
    with _lectures_lock:
        return {
            "ok": True,
            "lectures": [
                {
                    "lecture_id": k,
                    "running": v.running,
                    "in_queue": v.in_queue,
                    "in_amqp_url": v.in_amqp_url,
                    "out_queue": v.out_queue,
                    "last_error": v.last_error,
                }
                for k, v in _lectures.items()
            ],
        }


@app.get("/api/lecture/last_result/{lecture_id}")
def lecture_last_result(lecture_id: str):
    with _last_results_lock:
        return {"ok": True, "lecture_id": lecture_id, "result": _last_results.get(lecture_id)}


@app.get("/health")
def health():
    return {"ok": True}
