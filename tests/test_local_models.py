"""Opt-in real ONNX checks: FACE_TEST_MODEL_ROOT=.models python -m pytest -q."""
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
from insightface.app import FaceAnalysis
from skimage import data

from face_service.core import retina_embending as models

ROOT = os.getenv("FACE_TEST_MODEL_ROOT")
pytestmark = pytest.mark.skipif(not ROOT, reason="Set FACE_TEST_MODEL_ROOT to downloaded model root")


def test_official_recognizer_weights_identical():
    hashes = []
    for name in ("buffalo_l", "buffalo_m"):
        path = Path(ROOT) / "models" / name / "w600k_r50.onnx"
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    assert hashes[0] == hashes[1] == "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"


@pytest.mark.parametrize("name", ["buffalo_m", "buffalo_l"])
def test_real_onnx_and_upstream_equivalence(monkeypatch, name):
    monkeypatch.setattr(models, "MODEL_ROOT", ROOT)
    monkeypatch.setattr(models, "MODEL_NAME", name)
    monkeypatch.setattr(models, "_app", None)
    monkeypatch.setattr(models, "_rec", None)
    app = models.get_app()
    assert set(app.models) == {"recognition", "detection"}
    for model in app.models.values():
        assert model.session.get_session_options().intra_op_num_threads == 1
        assert model.session.get_providers() == ["CPUExecutionProvider"]
    # Public-domain sample bundled with scikit-image, never uploaded anywhere.
    img = data.astronaut()[:, :, ::-1].copy()
    faces = models.get_face_embeddings(img)
    assert faces and faces[0]["embedding"].shape == (512,)
    best = max(faces, key=lambda f: max(0, f["bbox"][2]-f["bbox"][0]) * max(0, f["bbox"][3]-f["bbox"][1]))
    selected = models.get_face_embeddings(img, largest_only=True)
    assert selected[0]["bbox"] == best["bbox"]
    np.testing.assert_array_equal(selected[0]["embedding"], best["embedding"])
    upstream = FaceAnalysis(name=name, root=ROOT, allowed_modules=["detection", "recognition"],
                            providers=["CPUExecutionProvider"])
    upstream.prepare(ctx_id=-1, det_size=(models.DEFAULT_DET_SIZE, models.DEFAULT_DET_SIZE))
    reference = upstream.get(img)
    assert len(reference) == len(faces)
    for left, right in zip(reference, faces):
        np.testing.assert_allclose(left.bbox, right["bbox"], atol=1.1)
        assert np.dot(left.normed_embedding, right["embedding"]) > .99999


def test_official_nested_archive_layout(monkeypatch, tmp_path):
    nested = tmp_path / "buffalo_m"
    nested.mkdir()
    for filename in ("det_2.5g.onnx", "w600k_r50.onnx"):
        (nested / filename).symlink_to((Path(ROOT) / "models" / "buffalo_m" / filename).resolve())
    monkeypatch.setattr(models, "MODEL_NAME", "buffalo_m")
    monkeypatch.setattr(models, "ensure_available", lambda *a, **kw: str(tmp_path))
    app = models.ConfiguredFaceAnalysis()
    assert app.model_dir == str(nested)
    assert set(app.models) == {"detection", "recognition"}
