import concurrent.futures
import json
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api
from face_service.core import recognize, retina_embending as models, runtime
from logic import dataset_store


class MemoryRedis:
    def __init__(self):
        self.data = {}

    def set(self, key, value):
        self.data[key] = value

    def get(self, key):
        return self.data.get(key)

    def delete(self, key):
        return int(self.data.pop(key, None) is not None)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "get_app", lambda: object())
    monkeypatch.setattr(api, "rdb", MemoryRedis())
    monkeypatch.setattr(api, "_lectures", {})
    with TestClient(api.app) as value:
        yield value


def test_http_and_redis_contracts(client, monkeypatch):
    assert client.get("/health").json() == {"ok": True}
    payload = {"lecture_id": 12, "persons": [{"person_id": "123", "embedding": [1., 0.]}]}
    assert client.post("/dataset", json=payload).json() == {
        "ok": True, "lecture_id": 12, "persons_count": 1}
    assert api.rdb.get("persons:12").startswith(b"NPZ1:")
    np.testing.assert_array_equal(dataset_store.load_dataset(api.rdb, 12)["123"][0], [1., 0.])
    # Legacy Redis JSON is still readable.
    api.rdb.set("persons:13", b'{"123": [[0.0, 1.0]]}')
    np.testing.assert_array_equal(dataset_store.load_dataset(api.rdb, 13)["123"][0], [0., 1.])
    monkeypatch.setattr(api, "best_embedding_bytes", lambda body: {
        "embedding": np.array([1., 0.], dtype=np.float32), "bbox": [1, 2, 3, 4]})
    assert client.post("/api/embedding", content=b"photo").json() == {
        "ok": True, "embedding": [1., 0.], "bbox": [1., 2., 3., 4.]}
    assert client.post("/api/embedding", content=b"").status_code == 400
    monkeypatch.setattr(api, "best_embedding_bytes", lambda body: None)
    assert client.post("/api/embedding", content=b"photo").status_code == 404
    assert client.delete("/dataset/12").json() == {"ok": True, "lecture_id": 12, "deleted": True}


def test_repeated_start_skips_external_work(client, monkeypatch):
    api._lectures[12] = SimpleNamespace(running=True, in_queue="old.in", out_queue="old.out")
    def fail(*args, **kwargs):
        pytest.fail("Repeated start must not access broker/backend/Redis")
    monkeypatch.setattr(api, "_test_rabbit_connect", fail)
    monkeypatch.setattr(api.requests, "get", fail)
    monkeypatch.setattr(api, "save_dataset", fail)
    assert client.post("/api/lecture/start", json={
        "lecture_id": 12, "in_amqp_url": "amqp://unused", "in_queue": "in"}).json() == {
        "ok": True, "status": "already_running", "lecture_id": 12,
        "in_queue": "old.in", "out_queue": "old.out"}


def test_embedding_does_not_block_event_loop(client, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def slow(body):
        entered.set()
        assert release.wait(5)
        return None
    monkeypatch.setattr(api, "best_embedding_bytes", slow)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        pending = pool.submit(client.post, "/api/embedding", content=b"photo")
        try:
            assert entered.wait(5)
            health = pool.submit(client.get, "/health")
            assert health.result(timeout=2).json() == {"ok": True}
        finally:
            release.set()
        assert pending.result().status_code == 404


def test_global_budget_shared_by_recognition_and_enrollment(monkeypatch):
    monkeypatch.setattr(runtime, "_inference_slots", threading.BoundedSemaphore(1))
    first_entered, release, second_entered = threading.Event(), threading.Event(), threading.Event()
    def recognition(*args):
        first_entered.set()
        assert release.wait(5)
        return []
    def enrollment(*args, **kwargs):
        second_entered.set()
        return []
    monkeypatch.setattr(recognize, "_recognize_bytes", recognition)
    monkeypatch.setattr(recognize, "_embeddings_bytes", enrollment)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        first = pool.submit(recognize.recognize_bytes, b"photo", {})
        assert first_entered.wait(5)
        second = pool.submit(recognize.embeddings_bytes, b"photo")
        try:
            assert not second_entered.wait(0.1)
        finally:
            release.set()
        assert first.result() == second.result() == []
        assert second_entered.is_set()


def test_budget_released_after_exception(monkeypatch):
    monkeypatch.setattr(runtime, "_inference_slots", threading.BoundedSemaphore(1))
    with pytest.raises(ValueError), runtime.inference_slot():
        raise ValueError("decode failed")
    assert runtime._inference_slots.acquire(blocking=False)
    runtime._inference_slots.release()


def test_model_initialization_is_atomic_and_once(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    class FakeApp:
        def __init__(self):
            calls.append(1)
            self.det_model = SimpleNamespace(detect=lambda img: None)
            self.models = {"recognition": SimpleNamespace(
                input_size=(112, 112), get_feat=lambda img: None,
                session=SimpleNamespace(get_session_options=lambda: SimpleNamespace(intra_op_num_threads=1)))}
        def prepare(self, **kwargs):
            entered.set()
            assert release.wait(5)
    monkeypatch.setattr(models, "_app", None)
    monkeypatch.setattr(models, "_rec", None)
    monkeypatch.setattr(models, "ConfiguredFaceAnalysis", FakeApp)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(models.get_app) for _ in range(8)]
        try:
            assert entered.wait(5)
            assert models._app is None and models._rec is None
        finally:
            release.set()
        results = [job.result() for job in jobs]
    assert len(calls) == 1
    assert all(app is results[0] for app in results)


def test_failed_initialization_can_retry(monkeypatch):
    attempts = []
    def fail():
        attempts.append(1)
        raise RuntimeError("missing weights")
    monkeypatch.setattr(models, "_app", None)
    monkeypatch.setattr(models, "ConfiguredFaceAnalysis", fail)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="missing weights"):
            models.get_app()
        assert models._app is None
    assert len(attempts) == 2


def test_largest_face_selected_before_embedding(monkeypatch):
    calls = []
    boxes = np.array([[0., 0., 10., 10., .9], [1., 1., 21., 21., .8], [2., 2., 22., 22., .7]])
    kps = np.zeros((3, 5, 2), dtype=np.float32)
    def embed(img, face):
        calls.append(face.bbox.copy())
        face.embedding = np.array([1., 0.], dtype=np.float32)
    app = SimpleNamespace(det_model=SimpleNamespace(detect=lambda *a, **kw: (boxes, kps)),
                          models={"recognition": SimpleNamespace(get=embed)})
    monkeypatch.setattr(models, "get_app", lambda: app)
    result = models.get_face_embeddings(np.zeros((30, 30, 3), np.uint8), largest_only=True)
    assert len(calls) == 1
    assert result[0]["bbox"] == [1, 1, 21, 21]  # First maximum, same as old max().
    np.testing.assert_array_equal(result[0]["embedding"], [1., 0.])


@pytest.mark.parametrize("size,expected_sides", [(150, [1200]), (250, [1000]), (400, [800, 1600]), (700, [700])])
def test_existing_upscale_and_fallback_preserved(monkeypatch, size, expected_sides):
    monkeypatch.setattr(recognize, "DIRECT_RECOGNITION", False)
    seen = []
    def detect(img, *args, **kwargs):
        seen.append(img.shape[0])
        return []
    monkeypatch.setattr(recognize, "recognize_bgr", detect)
    ok, jpg = cv2.imencode(".jpg", np.zeros((size, size, 3), np.uint8))
    assert ok
    assert recognize.recognize_bytes(jpg.tobytes(), {}) == []
    assert seen == expected_sides


def test_prebuilt_index_works_without_source_gallery(monkeypatch):
    persons = {"alice": [np.array([1., 0.])], "bob": [np.array([0., 1.])]}
    index = recognize.build_persons_index(persons)
    monkeypatch.setattr(recognize, "DIRECT_RECOGNITION", False)
    monkeypatch.setattr(recognize, "get_face_embeddings", lambda img: [
        {"embedding": np.array([1., 0.]), "bbox": [0, 0, 10, 10], "det_score": .9}])
    img = np.zeros((10, 10, 3), np.uint8)
    assert recognize.recognize_bgr(img, {}, prebuilt_index=index) == recognize.recognize_bgr(img, persons)


@pytest.mark.parametrize("binary", [False, True])
def test_amqp_input_output_and_ack_contract(monkeypatch, binary):
    events = []
    rt = api.LectureRuntime(12, "amqp://in", "faces.in", "amqp://out", "faces.out", .45)
    monkeypatch.setattr(api, "load_dataset", lambda *a: {"123": [np.array([1., 0.])]})
    monkeypatch.setattr(api, "notify_backend_start", lambda *a: None)
    monkeypatch.setattr(api, "notify_backend_stop", lambda *a: None)
    def recognize_input(data, persons, threshold, prebuilt_index):
        assert data == (b"jpeg" if binary else "anBlZw==")
        assert persons == {}
        assert threshold == .6
        np.testing.assert_array_equal(prebuilt_index[0], [[1., 0.]])
        return [{"person_id": "123", "matched": True, "score": .9}]
    monkeypatch.setattr(api, "recognize_bytes" if binary else "recognize_b64", recognize_input)
    class Channel:
        def queue_declare(self, **kwargs):
            pass
        def basic_qos(self, **kwargs):
            pass
        def basic_consume(self, **kwargs):
            self.callback = kwargs["on_message_callback"]
        def start_consuming(self):
            props = SimpleNamespace(content_type="image/jpeg" if binary else "application/json",
                                    headers={"threshold": .6})
            body = b"jpeg" if binary else json.dumps({"image_b64": "anBlZw==", "threshold": .6}).encode()
            self.callback(self, SimpleNamespace(delivery_tag=7), props, body)
        def basic_publish(self, **kwargs):
            events.append(("publish", kwargs["routing_key"], json.loads(kwargs["body"])))
        def basic_ack(self, **kwargs):
            events.append(("ack", kwargs["delivery_tag"]))
    class Connection:
        is_closed = False
        def channel(self):
            return Channel()
        def add_callback_threadsafe(self, callback):
            callback()
        def close(self):
            self.is_closed = True
    monkeypatch.setattr(api, "_make_blocking_conn", lambda *a: Connection())
    api.run_lecture_consumer(rt)
    assert events == [("publish", "faces.out", {"lecture_id": 12, "person_id": "123"}), ("ack", 7)]
    assert not rt.running
