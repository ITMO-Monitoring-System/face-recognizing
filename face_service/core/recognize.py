from __future__ import annotations

from pathlib import Path
from typing import Any
import base64

import numpy as np
import cv2

from .retina_embending import get_face_embeddings


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(a @ b)


def recognize_bgr(img_bgr: np.ndarray, persons: dict[str, list[np.ndarray]], threshold: float = 0.45) -> list[dict[str, Any]]:
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


def recognize_image(image_path: str | Path, persons: dict[str, list[np.ndarray]], threshold: float = 0.45):
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    return recognize_bgr(img, persons, threshold=threshold)


def recognize_bytes(img_bytes: bytes, persons: dict[str, list[np.ndarray]], threshold: float = 0.45):
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []
    return recognize_bgr(img, persons, threshold=threshold)


def recognize_b64(image_b64: str, persons: dict[str, list[np.ndarray]], threshold: float = 0.45):
    img_bytes = base64.b64decode(image_b64)
    return recognize_bytes(img_bytes, persons, threshold=threshold)

def embeddings_bytes(
    img_bytes: bytes,
    face_app,
) -> list[dict[str, Any]]:

    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    return get_face_embeddings(img, face_app)



def best_embedding_bytes(
    img_bytes: bytes,
    face_app,
) -> dict[str, Any] | None:

    faces = embeddings_bytes(img_bytes, face_app)
    if not faces:
        return None

    def area(face):
        x1, y1, x2, y2 = face["bbox"]
        return float(max(0, x2 - x1) * max(0, y2 - y1))

    return max(faces, key=area)