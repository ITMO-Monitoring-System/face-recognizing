import logging
import os

import cv2
import numpy as np
from insightface.app import FaceAnalysis

log = logging.getLogger(__name__)

_app: FaceAnalysis | None = None
DEFAULT_CTX_ID = int(os.getenv("FACE_ANALYSIS_CTX_ID", "-1"))
DEFAULT_DET_SIZE = int(os.getenv("FACE_ANALYSIS_DET_SIZE", "640"))


def get_app() -> FaceAnalysis:
    global _app
    if _app is None:
        _app = FaceAnalysis(allowed_modules=["detection", "recognition"])
        try:
            _app.prepare(ctx_id=DEFAULT_CTX_ID, det_size=(DEFAULT_DET_SIZE, DEFAULT_DET_SIZE))
            log.info(
                "InsightFace prepared with ctx_id=%s det_size=%s",
                DEFAULT_CTX_ID,
                DEFAULT_DET_SIZE,
            )
        except Exception:
            if DEFAULT_CTX_ID == -1:
                raise
            log.exception("Failed to initialize GPU ctx_id=%s, falling back to CPU", DEFAULT_CTX_ID)
            _app.prepare(ctx_id=-1, det_size=(DEFAULT_DET_SIZE, DEFAULT_DET_SIZE))
    return _app


def get_face_embeddings(img_bgr: np.ndarray) -> list[dict[str, object]]:
    app = get_app()
    faces = app.get(img_bgr)
    out: list[dict[str, object]] = []
    for f in faces:
        emb = f.normed_embedding.astype(np.float32)
        bbox = f.bbox.astype(int).tolist()
        kps = f.kps.astype(float).tolist()
        out.append({"embedding": emb, "bbox": bbox, "kps": kps, "det_score": float(f.det_score)})
    return out


def embed_crop_direct(img_bgr: np.ndarray) -> list[dict[str, object]]:
    """Fast path: input is an already-cropped face, skip detection and run ArcFace only.

    Used when face-tracking already detected and cropped the face (via MediaPipe).
    Avoids the double-detection (MediaPipe → InsightFace detector) penalty.
    """
    app = get_app()
    rec = app.models.get("recognition") if hasattr(app, "models") else None
    if rec is None:
        # Fallback: full pipeline if recognition model not accessible as expected
        return get_face_embeddings(img_bgr)

    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return []

    input_size = getattr(rec, "input_size", (112, 112))
    face_112 = cv2.resize(img_bgr, input_size)

    raw = rec.get_feat(face_112)
    emb = np.asarray(raw, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(emb))
    normed = emb / (norm + 1e-9)

    return [
        {
            "embedding": normed.astype(np.float32),
            "bbox": [0, 0, int(w), int(h)],
            "kps": [],
            "det_score": 1.0,
        }
    ]
