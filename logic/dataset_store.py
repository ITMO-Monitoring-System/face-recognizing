from __future__ import annotations
import os
import redis

import json
from typing import Dict, List, Optional

import numpy as np
import redis


def make_redis() -> redis.Redis:
    # пример: redis://redis:6379/0
    # rurl = "redis://localhost:6379/0"
    # return redis.Redis.from_url(rurl, decode_responses=True)
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return redis.Redis.from_url(url, decode_responses=True)


def key_persons(lecture_id: int | str) -> str:
    return f"persons:{lecture_id}"


def save_dataset(r: redis.Redis, lecture_id: str, persons: Dict[str, List[np.ndarray]]) -> None:
    # сериализуем np.ndarray -> list[float]
    raw = {pid: [emb.astype(float).tolist() for emb in embs] for pid, embs in persons.items()}
    r.set(key_persons(lecture_id), json.dumps(raw, ensure_ascii=False))


def load_dataset(r: redis.Redis, lecture_id: str) -> Optional[Dict[str, List[np.ndarray]]]:
    s = r.get(key_persons(lecture_id))
    if not s:
        return None
    raw = json.loads(s)
    persons: Dict[str, List[np.ndarray]] = {}
    for pid, emb_lists in raw.items():
        persons[pid] = [np.array(e, dtype=np.float32) for e in emb_lists]
    return persons


def delete_dataset(r: redis.Redis, lecture_id: str) -> bool:
    return bool(r.delete(key_persons(lecture_id)))
