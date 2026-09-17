import logging
import os
import threading
from pathlib import Path

# Multi-worker ONNX inference: pin native thread pools to 1 thread per worker.
# Без этого ORT/BLAS по умолчанию берут все ядра CPU на один инференс, и при
# FACE_RECOGNITION_WORKERS=N получается N×N тредов конкурируют за N ядер.
# Должно быть выставлено ДО импорта insightface (он подтягивает onnxruntime).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.model_zoo.model_zoo import ModelRouter
from insightface.utils.storage import ensure_available

log = logging.getLogger(__name__)

_app: FaceAnalysis | None = None
_rec = None
_input_size: tuple[int, int] = (112, 112)
_app_lock = threading.Lock()

DEFAULT_CTX_ID = int(os.getenv("FACE_ANALYSIS_CTX_ID", "-1"))
DEFAULT_DET_SIZE = int(os.getenv("FACE_ANALYSIS_DET_SIZE", "640"))
DIRECT_RECOGNITION = os.getenv("FACE_DIRECT_RECOGNITION", "0") == "1"
MODEL_NAME = os.getenv("FACE_ANALYSIS_MODEL", "buffalo_m")
MODEL_ROOT = os.getenv("FACE_ANALYSIS_ROOT", "~/.insightface")


class ConfiguredFaceAnalysis(FaceAnalysis):
    """Keep upstream inference; load only required models with explicit ORT options.

    InsightFace 0.7.3 model_zoo.get_model drops sess_options, whereas its
    ModelRouter forwards them. This adapter is tested against that pinned version.
    """

    def __init__(self):
        detectors = {"buffalo_m": "det_2.5g.onnx", "buffalo_l": "det_10g.onnx"}
        if MODEL_NAME not in detectors:
            raise ValueError("FACE_ANALYSIS_MODEL must be buffalo_m or buffalo_l")
        self.model_dir = ensure_available("models", MODEL_NAME, root=MODEL_ROOT)
        # The official buffalo_m ZIP contains a buffalo_m/ directory, while
        # buffalo_l ZIP stores ONNX files at its root. Support both layouts.
        nested = Path(self.model_dir) / MODEL_NAME
        if not (Path(self.model_dir) / detectors[MODEL_NAME]).is_file() and nested.is_dir():
            self.model_dir = str(nested)
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(os.getenv("FACE_ORT_THREADS", "1")))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        providers = ["CPUExecutionProvider"]
        if DEFAULT_CTX_ID >= 0:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.models = {}
        for task, filename in (("detection", detectors[MODEL_NAME]),
                               ("recognition", "w600k_r50.onnx")):
            path = Path(self.model_dir) / filename
            if not path.is_file():
                raise FileNotFoundError(f"Incomplete {MODEL_NAME} pack: {path}")
            model = ModelRouter(str(path)).get_model(providers=providers, sess_options=options)
            if model is None or model.taskname != task:
                raise RuntimeError(f"Unexpected model in {path}: expected {task}")
            self.models[task] = model
        self.det_model = self.models["detection"]


def get_app() -> FaceAnalysis:
    global _app, _rec, _input_size
    if _app is not None:
        return _app
    with _app_lock:
        if _app is not None:
            return _app
        candidate = ConfiguredFaceAnalysis()
        try:
            candidate.prepare(ctx_id=DEFAULT_CTX_ID, det_size=(DEFAULT_DET_SIZE, DEFAULT_DET_SIZE))
        except Exception:
            if DEFAULT_CTX_ID == -1:
                raise
            log.exception("Failed to initialize GPU ctx_id=%s, falling back to CPU", DEFAULT_CTX_ID)
            candidate.prepare(ctx_id=-1, det_size=(DEFAULT_DET_SIZE, DEFAULT_DET_SIZE))
        rec = candidate.models["recognition"]
        size = rec.input_size
        candidate.det_model.detect(np.zeros((DEFAULT_DET_SIZE, DEFAULT_DET_SIZE, 3), dtype=np.uint8))
        rec.get_feat(np.zeros((size[1], size[0], 3), dtype=np.uint8))
        _rec, _input_size = rec, size
        _app = candidate
        log.info("InsightFace ready: model=%s det_size=%s direct=%s ort_threads=%s",
                 MODEL_NAME, DEFAULT_DET_SIZE, DIRECT_RECOGNITION,
                 rec.session.get_session_options().intra_op_num_threads)
    return _app


def get_face_embeddings(img_bgr: np.ndarray, *, largest_only: bool = False) -> list[dict[str, object]]:
    app = get_app()
    if largest_only:
        bboxes, kpss = app.det_model.detect(img_bgr, max_num=0, metric="default")
        if len(bboxes) == 0:
            return []
        # Same integer area and first-on-tie rule as best_embedding_bytes.
        boxes = bboxes[:, :4].astype(int)
        areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
        idx = int(np.argmax(areas))
        face = Face(bbox=bboxes[idx, :4], kps=kpss[idx] if kpss is not None else None,
                    det_score=bboxes[idx, 4])
        app.models["recognition"].get(img_bgr, face)
        faces = [face]
    else:
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
