from __future__ import annotations

from pathlib import Path
from typing import Any
import base64

import logging
import numpy as np
import cv2

from .retina_embending import get_face_embeddings

log = logging.getLogger(__name__)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(a @ b)


def recognize_bgr(img_bgr: np.ndarray, persons: dict[str, list[np.ndarray]], threshold: float = 0.3) -> list[dict[str, Any]]:
    faces = get_face_embeddings(img_bgr)
    if not faces:
        return []

    results: list[dict[str, Any]] = []
    for face in faces:
        emb = face["embedding"]

        best_id, best_score = None, -1.0
        for pid, embs in persons.items():
            score = max(cosine_sim(emb, e) for e in embs)
            if score > best_score:
                best_id, best_score = pid, score

        match = best_score >= threshold
        results.append({
            "bbox": face["bbox"],
            "person_id": best_id if match else None,
            "score": round(best_score, 4),
            "matched": match,
        })

    return results


def recognize_image(image_path: str | Path, persons: dict[str, list[np.ndarray]], threshold: float = 0.3):
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    return recognize_bgr(img, persons, threshold=threshold)


def recognize_bytes(img_bytes: bytes, persons: dict[str, list[np.ndarray]], threshold: float = 0.3):
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    h, w = img.shape[:2]

    # 1) базовый upscale для маленьких кадров (поднимает шанс детекта)
    scale = 1.0
    if min(w, h) < 600:
        scale = 2.0
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    res = recognize_bgr(img, persons, threshold=threshold)

    # 2) fallback retry: если не нашли лицо и исходник был маленький — пробуем ещё сильнее
    # (иногда 4x даёт детект там, где 2x не даёт)
    if not res and min(w, h) < 600 and scale < 4.0:
        img2 = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)  # итого 4x от исходника
        res = recognize_bgr(img2, persons, threshold=threshold)

    # 3) debug лог (оставь хотя бы на время)
    try:
        log.info("recognize_bytes: orig=%sx%s scale=%.1f res=%s", w, h, scale, len(res))
    except Exception:
        pass

    return res



def recognize_b64(image_b64: str, persons: dict[str, list[np.ndarray]], threshold: float = 0.3):
    if image_b64 and image_b64.strip().lower().startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]  # убрать "data:image/...;base64,"
    img_bytes = base64.b64decode(image_b64, validate=False)
    return recognize_bytes(img_bytes, persons, threshold=threshold)

def embeddings_bytes(img_bytes: bytes) -> list[dict[str, Any]]:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    h, w = img.shape[:2]
    if min(w, h) < 600:
        img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    faces = get_face_embeddings(img)
    if not faces and min(w, h) < 600:
        img2 = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        faces = get_face_embeddings(img2)

    return faces


def best_embedding_bytes(img_bytes: bytes) -> dict[str, Any] | None:
    faces = embeddings_bytes(img_bytes)
    if not faces:
        return None

    def area(face: dict[str, Any]) -> float:
        x1, y1, x2, y2 = face["bbox"]
        return float(max(0, x2 - x1) * max(0, y2 - y1))

    return max(faces, key=area)