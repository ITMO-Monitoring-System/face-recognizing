"""Process-wide CPU budget shared by AMQP lectures and the embedding API."""
import os
import threading
from contextlib import contextmanager

# Must precede numpy/InsightFace imports when running outside Docker, too.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")


def _default_workers() -> int:
    return max(1, min(4, (os.cpu_count() or 2) // 2))


INFERENCE_WORKERS = max(1, int(os.getenv("FACE_RECOGNITION_WORKERS", str(_default_workers()))))
_inference_slots = threading.BoundedSemaphore(INFERENCE_WORKERS)


@contextmanager
def inference_slot():
    # Bound decode/upscaling buffers as well as ONNX inference. Consumers retain
    # their existing IO threads, prefetch and shutdown/ack ownership.
    with _inference_slots:
        yield
