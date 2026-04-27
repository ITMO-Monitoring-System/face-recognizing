from __future__ import annotations
import io
import json
import os
from typing import Dict, List, Optional

import numpy as np
import redis


# Магический префикс для бинарного формата (npz), позволяет отличить от старого JSON
_BIN_MAGIC = b"NPZ1:"


def make_redis() -> redis.Redis:
    """Redis-клиент в бинарном режиме (decode_responses=False).

    Эмбеддинги храним как numpy npz — ~4x компактнее JSON и быстрее на десериализации.
    Все строковые ключи Redis работают и в бинарном режиме (redis-py принимает str).
    """
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return redis.Redis.from_url(url, decode_responses=False)


def key_persons(lecture_id: int | str) -> str:
    return f"persons:{int(lecture_id)}"


def save_dataset(r: redis.Redis, lecture_id: int, persons: Dict[str, List[np.ndarray]]) -> None:
    arrays: Dict[str, np.ndarray] = {}
    for pid, embs in persons.items():
        for i, emb in enumerate(embs):
            # Ключ "pid__i". Сам pid не должен содержать "__" — на практике это student_id.
            arrays[f"{pid}__{i}"] = np.asarray(emb, dtype=np.float32)

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    r.set(key_persons(lecture_id), _BIN_MAGIC + buf.getvalue())


def load_dataset(r: redis.Redis, lecture_id: int) -> Optional[Dict[str, List[np.ndarray]]]:
    data = r.get(key_persons(lecture_id))
    if not data:
        return None

    # Бинарный формат (новый)
    if isinstance(data, (bytes, bytearray)) and data.startswith(_BIN_MAGIC):
        buf = io.BytesIO(bytes(data[len(_BIN_MAGIC):]))
        with np.load(buf, allow_pickle=False) as npz:
            persons: Dict[str, List[np.ndarray]] = {}
            # Сохраняем порядок индексов внутри каждого pid
            grouped: Dict[str, List[tuple[int, np.ndarray]]] = {}
            for key in npz.files:
                pid, idx_s = key.rsplit("__", 1)
                grouped.setdefault(pid, []).append((int(idx_s), npz[key].astype(np.float32)))
            for pid, items in grouped.items():
                items.sort(key=lambda t: t[0])
                persons[pid] = [arr for _, arr in items]
        return persons

    # Старый JSON-формат (на случай, если в Redis остались данные от прежней версии)
    if isinstance(data, (bytes, bytearray)):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return None
    else:
        text = data
    try:
        raw = json.loads(text)
    except (ValueError, TypeError):
        return None
    persons = {}
    for pid, emb_lists in raw.items():
        persons[pid] = [np.array(e, dtype=np.float32) for e in emb_lists]
    return persons


def delete_dataset(r: redis.Redis, lecture_id: int) -> bool:
    return bool(r.delete(key_persons(lecture_id)))
