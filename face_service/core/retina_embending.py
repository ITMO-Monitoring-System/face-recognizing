import cv2
import numpy as np
from typing import Any
from insightface.app import FaceAnalysis

_app: FaceAnalysis | None = None


def get_app() -> FaceAnalysis:
    global _app
    if _app is None:
        _app = FaceAnalysis(allowed_modules=["detection", "recognition"])
        _app.prepare(ctx_id=-1, det_size=(640, 640))
    return _app


def get_face_embeddings(
    img_bgr: np.ndarray,
    face_app,
) -> list[dict[str, Any]]:

    faces = face_app.get(img_bgr)
    result = []

    for f in faces:
        result.append({
            "embedding": f.embedding,
            "bbox": f.bbox.tolist(),
        })

    return result
