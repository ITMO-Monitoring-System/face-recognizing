import cv2
import numpy as np
from insightface.app import FaceAnalysis

_app: FaceAnalysis | None = None


def get_app() -> FaceAnalysis:
    global _app
    if _app is None:
        _app = FaceAnalysis(allowed_modules=["detection", "recognition"])
        _app.prepare(ctx_id=-1, det_size=(1024, 1024))
    return _app


def get_face_embeddings(img_bgr: np.ndarray):
    app = get_app()
    faces = app.get(img_bgr)
    out = []
    for f in faces:
        emb = f.normed_embedding.astype(np.float32)
        bbox = f.bbox.astype(int).tolist()
        kps = f.kps.astype(float).tolist()
        out.append({"embedding": emb, "bbox": bbox, "kps": kps, "det_score": float(f.det_score)})
    return out