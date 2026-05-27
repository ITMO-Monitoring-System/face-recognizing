import logging
import os

# Multi-worker ONNX inference: pin native thread pools to 1 thread per worker.
# Без этого ORT/BLAS по умолчанию берут все ядра CPU на один инференс, и при
# FACE_RECOGNITION_WORKERS=N получается N×N тредов конкурируют за N ядер.
# Должно быть выставлено ДО импорта insightface (он подтягивает onnxruntime).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2
import numpy as np
from insightface.app import FaceAnalysis

log = logging.getLogger(__name__)

_app: FaceAnalysis | None = None
_rec = None
_input_size: tuple[int, int] = (112, 112)

DEFAULT_CTX_ID = int(os.getenv("FACE_ANALYSIS_CTX_ID", "-1"))
DEFAULT_DET_SIZE = int(os.getenv("FACE_ANALYSIS_DET_SIZE", "640"))
DIRECT_RECOGNITION = os.getenv("FACE_DIRECT_RECOGNITION", "0") == "1"


def get_app() -> FaceAnalysis:
    global _app, _rec, _input_size
    if _app is None:
        # ВАЖНО: insightface.FaceAnalysis имеет жёсткий assert "'detection' in self.models",
        # поэтому нельзя ограничить модули только recognition — иначе __init__ падает с
        # AssertionError и _app остаётся None, цикл повторяется на каждом кадре.
        # Detection-модель грузим всегда (~170MB), но при DIRECT_RECOGNITION её
        # никогда не вызываем — горячий путь идёт через embed_crop_direct.
        _app = FaceAnalysis(allowed_modules=["detection", "recognition"])
        try:
            _app.prepare(ctx_id=DEFAULT_CTX_ID, det_size=(DEFAULT_DET_SIZE, DEFAULT_DET_SIZE))
            log.info(
                "InsightFace prepared with ctx_id=%s det_size=%s direct=%s",
                DEFAULT_CTX_ID,
                DEFAULT_DET_SIZE,
                DIRECT_RECOGNITION,
            )
        except Exception:
            if DEFAULT_CTX_ID == -1:
                raise
            log.exception("Failed to initialize GPU ctx_id=%s, falling back to CPU", DEFAULT_CTX_ID)
            _app.prepare(ctx_id=-1, det_size=(DEFAULT_DET_SIZE, DEFAULT_DET_SIZE))
        _rec = _app.models.get("recognition") if hasattr(_app, "models") else None
        if _rec is not None:
            _input_size = getattr(_rec, "input_size", (112, 112))
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
    get_app()
    if _rec is None:
        return get_face_embeddings(img_bgr)

    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return []

    face_112 = cv2.resize(img_bgr, _input_size)

    raw = _rec.get_feat(face_112)
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
