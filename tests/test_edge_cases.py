# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import numpy as np
import pytest

from uniface.attribute import AgeGender, Emotion, FaceAttribNet, FairFace
from uniface.face_utils import compute_similarity, estimate_norm
from uniface.landmark import FaceMesh, Landmark106, PIPNet, roi_from_box
from uniface.matting import MODNet, PPMatting
from uniface.quality import EDifFIQA
from uniface.recognition import ArcFace
from uniface.recognition.base import PreprocessConfig
from uniface.spoofing import MiniFASNet
from uniface.types import Face


@pytest.fixture
def image():
    return np.random.default_rng(0).integers(0, 256, size=(480, 640, 3), dtype=np.uint8)


def test_modnet_rejects_too_small_image_and_input_size():
    with pytest.raises(ValueError, match='32'):
        MODNet(input_size=16)

    with pytest.raises(ValueError, match='32'):
        PPMatting(max_short=16)

    matting = MODNet(providers=['CPUExecutionProvider'])
    # Previously crashed cv2.resize when min(H, W) < 32 <= input_size <= max(H, W)
    with pytest.raises(ValueError, match='32'):
        matting.predict(np.zeros((20, 600, 3), dtype=np.uint8))


def test_facemesh_accepts_box_only_faces_and_1d_bbox(image):
    mesher = FaceMesh(providers=['CPUExecutionProvider'])

    box_only = Face(
        bbox=np.array([100.0, 100.0, 300.0, 300.0], dtype=np.float32),
        confidence=0.95,
        landmarks=np.empty((0, 2), dtype=np.float32),
    )
    res_faces = mesher.predict(image, [box_only])
    assert len(res_faces) == 1
    assert res_faces[0].landmarks.shape == (mesher.num_landmarks, 3)

    res_1d = mesher.predict(image, bboxes=np.array([100.0, 100.0, 300.0, 300.0], dtype=np.float32))
    assert len(res_1d) == 1

    with pytest.raises(ValueError, match='Invalid bounding box'):
        roi_from_box(np.array([200.0, 200.0, 100.0, 100.0]))


def test_pipnet_handles_bottom_edge_and_rejects_negative_oob_box(image):
    pip = PIPNet(providers=['CPUExecutionProvider'])

    # Valid box overlapping bottom 2px of a 480p image (asymmetric +10% top shift previously pushed y1 > y2)
    lms = pip.get_landmarks(image, [100, 478, 200, 500])
    assert lms.shape == (pip.num_lms, 2)

    # Negative out-of-bounds box must raise ValueError instead of negative-slicing full background
    with pytest.raises(ValueError, match='overlap'):
        pip.get_landmarks(image, [-100, -100, -50, -50])


def test_fairface_and_minifasnet_reject_oob_and_degenerate_boxes(image):
    fairface = FairFace(providers=['CPUExecutionProvider'])
    spoofer = MiniFASNet(providers=['CPUExecutionProvider'])

    for bad_box in ([-100, -100, -50, -50], [700, 700, 800, 800], [100, 100, 100, 100]):
        with pytest.raises(ValueError):
            fairface.preprocess(image, bad_box)

    for bad_box in ([-500, -500, -450, -450], [100.1, 100.1, 100.1, 100.1], [200, 200, 100, 100]):
        with pytest.raises(ValueError):
            spoofer.predict(image, bad_box)

    # Sub-pixel positive box must not divide by zero
    res = spoofer.predict(image, [100.1, 100.1, 100.8, 100.8])
    assert 0.0 <= res.confidence <= 1.0


def test_landmark106_postprocess_does_not_mutate_input():
    lm = Landmark106(providers=['CPUExecutionProvider'])
    raw = np.zeros((212,), dtype=np.float32)
    raw_copy = raw.copy()
    transform = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    out = lm.postprocess(raw, transform)
    assert out.shape == (106, 2)
    assert np.array_equal(raw, raw_copy)


def test_recognizer_list_mean_scalar_std_and_optional_landmarks():
    rec = ArcFace(
        preprocessing=PreprocessConfig(input_mean=[127.5, 127.5, 127.5], input_std=127.5),
        providers=['CPUExecutionProvider'],
    )
    aligned = np.random.default_rng(5).integers(0, 256, (112, 112, 3), dtype=np.uint8)

    blob = rec.preprocess(aligned)
    assert blob.shape == (1, 3, 112, 112)
    assert blob.flags['C_CONTIGUOUS']

    emb1 = rec.get_normalized_embedding(aligned)
    emb2 = rec(aligned)
    assert emb1.shape == (512,)
    assert np.allclose(emb1, emb2)


def test_estimate_norm_and_compute_similarity_edge_cases():
    # List of 5 points is coerced cleanly
    pts_list = [[38.0, 51.0], [73.0, 51.0], [56.0, 71.0], [41.0, 92.0], [70.0, 92.0]]
    m, inv_m = estimate_norm(pts_list)
    assert m.shape == (2, 3) and inv_m.shape == (2, 3)

    # Degenerate all-zero landmarks raise ValueError instead of internal skimage error
    with pytest.raises(ValueError, match='degenerate'):
        estimate_norm(np.zeros((5, 2), dtype=np.float32))

    # Small-norm and zero-norm vectors in compute_similarity
    small = np.full(512, 1e-4, dtype=np.float32)
    assert compute_similarity(small, small) == pytest.approx(1.0, abs=1e-6)
    assert compute_similarity(np.zeros(512, dtype=np.float32), small) == 0.0

    # Batch arrays are rejected rather than silently flattened
    with pytest.raises(ValueError, match='single vectors'):
        compute_similarity(np.ones((2, 512), dtype=np.float32), np.ones((2, 512), dtype=np.float32))


@pytest.mark.parametrize('cls', [AgeGender, FaceAttribNet, FairFace, MiniFASNet, EDifFIQA])
def test_invalid_string_model_name_raises_value_error(cls):
    with pytest.raises(ValueError, match='Unknown model identifier'):
        cls(model_name='nonexistent_weights')  # type: ignore[arg-type]


def test_emotion_detach_and_non_square_input_size(image):
    torch = pytest.importorskip('torch')

    emotion = Emotion(device='cpu')
    landmarks = np.array(
        [[250.0, 200.0], [390.0, 200.0], [320.0, 280.0], [260.0, 350.0], [380.0, 350.0]],
        dtype=np.float32,
    )
    face = Face(bbox=np.array([220.0, 170.0, 420.0, 390.0]), confidence=0.95, landmarks=landmarks)
    pred = emotion.predict(image, face)
    assert face.emotion == pred.emotion
    assert 0.0 <= pred.confidence <= 1.0

    # Verify (H, W) -> (W, H) cv2.resize orientation in preprocess
    emotion.input_size = (112, 128)
    tensor = emotion.preprocess(image, landmarks)
    assert tuple(tensor.shape) == (1, 3, 112, 128)

    logits = torch.randn(1, 7, requires_grad=True)
    result = emotion.postprocess(logits)
    assert 0.0 <= result.confidence <= 1.0


def test_faiss_load_rejects_mismatched_dimension_and_metadata_count(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import uniface.stores.faiss as faiss_mod

    class DummyIndex:
        def __init__(self, d: int, ntotal: int = 0):
            self.d = d
            self.ntotal = ntotal

    fake_faiss = SimpleNamespace(
        IndexFlatIP=lambda d: DummyIndex(d, 0),
        read_index=lambda _: DummyIndex(128, 2),
    )
    monkeypatch.setattr(faiss_mod, '_import_faiss', lambda: fake_faiss)

    store = faiss_mod.FAISS(embedding_size=512, db_path=str(tmp_path))
    (tmp_path / 'faiss_index.bin').write_bytes(b'dummy')
    (tmp_path / 'metadata.json').write_text(json.dumps([{'id': 1}, {'id': 2}]), encoding='utf-8')

    with pytest.raises(RuntimeError, match='dimension'):
        store.load()

    fake_faiss.read_index = lambda _: DummyIndex(512, 3)
    with pytest.raises(RuntimeError, match='metadata length'):
        store.load()
