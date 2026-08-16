# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


from __future__ import annotations

import numpy as np
import pytest

from uniface.attribute import FaceAttribNet, FaceStateResult
from uniface.types import Face

ATTRIBUTE_NAMES = ('left_eye_open', 'right_eye_open', 'eyeglasses', 'mask', 'sunglasses')


def _make_face(bbox: list[int] | np.ndarray) -> Face:
    """Helper: build a minimal Face from a bounding box."""
    bbox = np.asarray(bbox)
    landmarks = np.zeros((5, 2), dtype=np.float32)
    return Face(bbox=bbox, confidence=0.99, landmarks=landmarks)


@pytest.fixture(scope='module')
def model():
    return FaceAttribNet()


@pytest.fixture
def mock_image():
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_face():
    return _make_face([100, 100, 300, 300])


def test_model_initialization(model):
    assert model is not None, 'FaceAttribNet model initialization failed.'


def test_prediction_output_format(model, mock_image, mock_face):
    result = model.predict(mock_image, mock_face)
    assert isinstance(result, FaceStateResult), f'Result should be FaceStateResult, got {type(result)}'
    for name in ATTRIBUTE_NAMES:
        value = getattr(result, name)
        assert isinstance(value, float), f'{name} should be float, got {type(value)}'


def test_probabilities_in_range(model, mock_image, mock_face):
    result = model.predict(mock_image, mock_face)
    for name, value in result.as_dict().items():
        assert 0.0 <= value <= 1.0, f'{name} should be in [0, 1], got {value}'


def test_as_dict_keys(model, mock_image, mock_face):
    result = model.predict(mock_image, mock_face)
    assert tuple(result.as_dict().keys()) == ATTRIBUTE_NAMES


def test_labels_thresholding(model, mock_image, mock_face):
    result = model.predict(mock_image, mock_face)
    assert result.labels(threshold=-0.1) == list(ATTRIBUTE_NAMES), 'All attributes should exceed a negative threshold'
    assert result.labels(threshold=1.1) == [], 'No attribute should exceed a threshold above 1'


def test_preprocess_output_shape(model, mock_image, mock_face):
    blob = model.preprocess(mock_image, mock_face.bbox)
    assert blob.shape == (1, 3, 128, 128), f'Blob shape should be (1, 3, 128, 128), got {blob.shape}'
    assert blob.dtype == np.float32, f'Blob dtype should be float32, got {blob.dtype}'
    assert blob.min() >= 0.0 and blob.max() <= 1.0, 'Blob values should be in [0, 1]'


def test_face_enrichment(model, mock_image, mock_face):
    """predict() must write the five state probabilities back to the Face object."""
    for name in ATTRIBUTE_NAMES:
        assert getattr(mock_face, name) is None

    result = model.predict(mock_image, mock_face)

    for name in ATTRIBUTE_NAMES:
        assert getattr(mock_face, name) == getattr(result, name)


def test_consistency(model, mock_image, mock_face):
    result1 = model.predict(mock_image, mock_face)
    result2 = model.predict(mock_image, mock_face)
    assert result1.as_dict() == result2.as_dict(), 'Same input should produce identical predictions'


def test_different_bbox_sizes(model, mock_image):
    test_bboxes = [
        [50, 50, 150, 150],
        [100, 100, 300, 300],
        [50, 50, 400, 400],
    ]

    for bbox in test_bboxes:
        face = _make_face(bbox)
        result = model.predict(mock_image, face)
        for name, value in result.as_dict().items():
            assert 0.0 <= value <= 1.0, f'{name} out of range for bbox {bbox}'


def test_different_image_sizes(model):
    test_sizes = [(480, 640, 3), (720, 1280, 3), (1080, 1920, 3)]
    face = _make_face([100, 100, 300, 300])

    for size in test_sizes:
        mock_image = np.random.randint(0, 255, size, dtype=np.uint8)
        result = model.predict(mock_image, face)
        for name, value in result.as_dict().items():
            assert 0.0 <= value <= 1.0, f'{name} out of range for image size {size}'


def test_non_overlapping_bbox_raises(model, mock_image):
    face = _make_face([700, 700, 800, 800])  # outside the 640x640 image
    with pytest.raises(ValueError, match='does not overlap'):
        model.predict(mock_image, face)


def test_margin_expands_crop(mock_image, mock_face, model):
    """A margin model should still produce valid probabilities."""
    margin_model = FaceAttribNet(margin=0.2, providers=model.providers)
    result = margin_model.predict(mock_image, mock_face)
    for name, value in result.as_dict().items():
        assert 0.0 <= value <= 1.0, f'{name} out of range with margin'


def test_result_repr(model, mock_image, mock_face):
    result = model.predict(mock_image, mock_face)
    text = repr(result)
    assert text.startswith('FaceStateResult(')
    for name in ATTRIBUTE_NAMES:
        assert name in text
