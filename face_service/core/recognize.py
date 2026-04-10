from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .retina_embending import get_face_embeddings

log = logging.getLogger(__name__)
DEFAULT_RECOGNITION_THRESHOLD = float(os.getenv("FACE_RECOGNITION_THRESHOLD", "0.55"))

def _build_persons_index(
    persons: dict[str, list[np.ndarray]],
) -> tuple[np.ndarray, list[str]] | tuple[None, None]:
    """Pre-stack all embeddings into a normalized matrix for fast matmul matching."""
    all_embs: list[np.ndarray] = []
    all_pids: list[str] = []
    for pid, embs in persons.items():
        for e in embs:
            norm = np.linalg.norm(e)
            all_embs.append(e / (norm + 1e-9))
            all_pids.append(pid)
    if not all_embs:
        return None, None
    return np.stack(all_embs).astype(np.float32), all_pids


def recognize_bgr(
    img_bgr: np.ndarray,
    persons: dict[str, list[np.ndarray]],
    threshold: float = DEFAULT_RECOGNITION_THRESHOLD,
) -> list[dict[str, Any]]:
    faces = get_face_embeddings(img_bgr)
    if not faces:
        return []

    # Build normalized embedding matrix once for all faces in this frame
    emb_matrix, all_pids = _build_persons_index(persons) if persons else (None, None)

    results: list[dict[str, Any]] = []
    for face in faces:
        emb = face["embedding"]

        if emb_matrix is None:
            best_id, best_score = None, -1.0
        else:
            # Vectorized cosine similarity: single matmul instead of Python loop
            emb_norm = emb / (np.linalg.norm(emb) + 1e-9)
            sims = emb_matrix @ emb_norm  # shape: (N,)

            # Aggregate: best score per person
            scores_by_pid: dict[str, float] = {}
            for i, pid in enumerate(all_pids):
                s = float(sims[i])
                if s > scores_by_pid.get(pid, -2.0):
                    scores_by_pid[pid] = s

            best_id = max(scores_by_pid, key=scores_by_pid.__getitem__)
            best_score = scores_by_pid[best_id]

        match = best_score >= threshold
        results.append(
            {
                "bbox": face["bbox"],
                "person_id": best_id if match else None,
                "confidence": round(best_score, 4),
                "score": round(best_score, 4),
                "matched": match,
                "det_score": round(float(face.get("det_score", 0.0)), 4),
            }
        )

    return results


def recognize_image(
    image_path: str | Path,
    persons: dict[str, list[np.ndarray]],
    threshold: float = DEFAULT_RECOGNITION_THRESHOLD,
) -> list[dict[str, Any]]:
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    return recognize_bgr(img, persons, threshold=threshold)


def recognize_bytes(
    img_bytes: bytes,
    persons: dict[str, list[np.ndarray]],
    threshold: float = DEFAULT_RECOGNITION_THRESHOLD,
) -> list[dict[str, Any]]:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    h, w = img.shape[:2]


    # 1) базовый upscale для маленьких кадров (поднимает шанс детекта)
    scale = 1.0
    min_side = min(w, h)
    if min_side < 160:
        scale = 8.0
    elif min_side < 300:
        scale = 4.0
    elif min_side < 600:
        scale = 2.0

    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    res = recognize_bgr(img, persons, threshold=threshold)

    # 2) fallback retry: если не нашли лицо и исходник был маленький — пробуем ещё сильнее
    # (иногда 4x даёт детект там, где 2x не даёт)
    if not res and min_side < 600 and scale < 4.0:
        img2 = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)  # итого 4x от исходника
        res = recognize_bgr(img2, persons, threshold=threshold)

    # 3) debug лог (оставь хотя бы на время)
    try:
        log.info("recognize_bytes: orig=%sx%s scale=%.1f res=%s", w, h, scale, len(res))
    except Exception:
        pass

    return res



def recognize_b64(
    image_b64: str,
    persons: dict[str, list[np.ndarray]],
    threshold: float = DEFAULT_RECOGNITION_THRESHOLD,
) -> list[dict[str, Any]]:
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
