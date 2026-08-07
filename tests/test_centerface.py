# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


from __future__ import annotations

import numpy as np
import pytest

from uniface.constants import CenterFaceWeights
from uniface.detection import CenterFace


@pytest.fixture
def centerface_model():
    return CenterFace(
        model_name=CenterFaceWeights.DEFAULT,
        confidence_threshold=0.35,
        nms_threshold=0.3,
    )


def test_model_initialization(centerface_model):
    assert centerface_model is not None, 'Model initialization failed.'


def test_invalid_input_size():
    with pytest.raises(ValueError, match='strictly positive'):
        CenterFace(input_size=(0, 480))


@pytest.mark.parametrize('size', [(480, 640), (519, 713), (720, 1280)])
def test_inference_shape_is_padded_to_multiple_of_32(centerface_model, size):
    """The FPN needs both sides divisible by 32, whatever the caller passes in."""
    image = np.zeros((*size, 3), dtype=np.uint8)
    resized, scale_w, scale_h = centerface_model._resize(image)

    assert resized.shape[0] % 32 == 0
    assert resized.shape[1] % 32 == 0
    assert scale_w == pytest.approx(resized.shape[1] / size[1])
    assert scale_h == pytest.approx(resized.shape[0] / size[0])


def test_large_input_is_capped_but_small_input_is_not_upscaled(centerface_model):
    """input_size bounds cost on huge images without touching ordinary frames."""
    detector = CenterFace(input_size=(640, 640))

    big, _, _ = detector._resize(np.zeros((2000, 4000, 3), dtype=np.uint8))
    assert big.shape[1] <= 640 + 31 and big.shape[0] <= 320 + 31

    # A 640x480 camera frame runs at its own resolution, not letterboxed to 640x640
    frame, scale_w, scale_h = detector._resize(np.zeros((480, 640, 3), dtype=np.uint8))
    assert frame.shape[:2] == (480, 640)
    assert (scale_w, scale_h) == (1.0, 1.0)


def test_native_input_size_never_rescales(centerface_model):
    """input_size=None reproduces upstream: native resolution, rounded up only."""
    detector = CenterFace(input_size=None)
    resized, _, _ = detector._resize(np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert resized.shape[:2] == (1088, 1920)


def test_inference_on_640x640_image(centerface_model):
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    faces = centerface_model.detect(mock_image)

    assert isinstance(faces, list), 'Detections should be a list.'

    for face in faces:
        # Face is a dataclass, check attributes exist
        assert hasattr(face, 'bbox'), "Each detection should have a 'bbox' attribute."
        assert hasattr(face, 'confidence'), "Each detection should have a 'confidence' attribute."
        assert hasattr(face, 'landmarks'), "Each detection should have a 'landmarks' attribute."

        bbox = face.bbox
        assert len(bbox) == 4, 'BBox should have 4 values (x1, y1, x2, y2).'

        landmarks = face.landmarks
        assert len(landmarks) == 5, 'Should have 5 landmark points.'
        assert all(len(pt) == 2 for pt in landmarks), 'Each landmark should be (x, y).'


def test_confidence_threshold(centerface_model):
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    faces = centerface_model.detect(mock_image)

    for face in faces:
        confidence = face.confidence
        assert confidence >= 0.35, f'Detection has confidence {confidence} below threshold 0.35'


def test_no_faces_detected(centerface_model):
    empty_image = np.zeros((640, 640, 3), dtype=np.uint8)
    faces = centerface_model.detect(empty_image)
    assert len(faces) == 0, 'Should detect no faces in a blank image.'


def test_different_input_sizes(centerface_model):
    test_sizes = [(480, 640, 3), (720, 1280, 3), (1080, 1920, 3)]

    for size in test_sizes:
        mock_image = np.random.randint(0, 255, size, dtype=np.uint8)
        faces = centerface_model.detect(mock_image)
        assert isinstance(faces, list), f'Should return list for size {size}'
