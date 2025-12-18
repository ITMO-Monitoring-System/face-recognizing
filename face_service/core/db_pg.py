from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2
import psycopg

from .retina_embending import get_face_embeddings


@dataclass(frozen=True)
class DbConfig:
    dsn: str  # postgresql://user:pass@host:port/dbname


def _bytea_to_bgr(img_bytes: bytes) -> Optional[np.ndarray]:

    if not img_bytes:
        return None
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _best_embedding(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    faces = get_face_embeddings(img_bgr)
    if not faces:
        return None
    best = max(faces, key=lambda f: f["det_score"])
    return best["embedding"].astype(np.float32)


def load_persons_from_db(cfg: DbConfig) -> dict[str, list[np.ndarray]]:
    """
    Возвращает:
      persons[isu] = [emb_left, emb_right, emb_full]
    """
    persons: dict[str, list[np.ndarray]] = {}

    query = """
    SELECT student_id, left_face, right_face, full_face
    FROM cores.face_images
    """

    with psycopg.connect(cfg.dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            for student_id, left_b, right_b, full_b in cur.fetchall():
                embs: list[np.ndarray] = []

                for blob in (left_b, right_b, full_b):
                    if blob is None:
                        continue
                    img = _bytea_to_bgr(blob)
                    if img is None:
                        continue
                    emb = _best_embedding(img)
                    if emb is None:
                        continue
                    embs.append(emb)

                if embs:
                    persons[str(student_id)] = embs

    return persons

def load_persons_from_pg_embeddings(cfg: DbConfig) -> dict[str, list[np.ndarray]]:
    """
    persons[isu] = [embedding]
    """
    persons: dict[str, list[np.ndarray]] = {}

    query = """
    SELECT student_id, embedding
    FROM cores.face_embeddings
    """

    with psycopg.connect(cfg.dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            for student_id, embedding_arr in cur.fetchall():
                # embedding_arr будет Python list[float]
                emb = np.array(embedding_arr, dtype=np.float32)
                persons.setdefault(str(student_id), []).append(emb)

    return persons

