import logging
import os

import numpy as np
from insightface.app import FaceAnalysis

log = logging.getLogger(__name__)

_app: FaceAnalysis | None = None
DEFAULT_CTX_ID = int(os.getenv("FACE_ANALYSIS_CTX_ID", "-1"))
DEFAULT_DET_SIZE = int(os.getenv("FACE_ANALYSIS_DET_SIZE", "1024"))


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
