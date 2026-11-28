# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


from __future__ import annotations

import numpy as np
import pytest

from uniface.constants import SCRFDWeights
from uniface.detection import SCRFD


@pytest.fixture
def scrfd_model():
    return SCRFD(
        model_name=SCRFDWeights.SCRFD_500M_KPS,
        confidence_threshold=0.5,
        nms_threshold=0.4,
    )


def test_model_initialization(scrfd_model):
    assert scrfd_model is not None, 'Model initialization failed.'


def test_inference_on_640x640_image(scrfd_model):
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    faces = scrfd_model.detect(mock_image)

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


def test_confidence_threshold(scrfd_model):
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    faces = scrfd_model.detect(mock_image)

    for face in faces:
        confidence = face.confidence
        assert confidence >= 0.5, f'Detection has confidence {confidence} below threshold 0.5'


def test_no_faces_detected(scrfd_model):
    empty_image = np.zeros((640, 640, 3), dtype=np.uint8)
    faces = scrfd_model.detect(empty_image)
    assert len(faces) == 0, 'Should detect no faces in a blank image.'


def test_different_input_sizes(scrfd_model):
    test_sizes = [(480, 640, 3), (720, 1280, 3), (1080, 1920, 3)]

    for size in test_sizes:
        mock_image = np.random.randint(0, 255, size, dtype=np.uint8)
        faces = scrfd_model.detect(mock_image)
        assert isinstance(faces, list), f'Should return list for size {size}'


def test_scrfd_10g_model():
    model = SCRFD(model_name=SCRFDWeights.SCRFD_10G_KPS, confidence_threshold=0.5)
    assert model is not None, 'SCRFD 10G model initialization failed.'

    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    faces = model.detect(mock_image)
    assert isinstance(faces, list), 'SCRFD 10G should return list of detections.'


def test_scrfd_preprocess_swaps_bgr_to_rgb_and_scales_by_128(scrfd_model):
    """SCRFD matches InsightFace's RGB (swapRB=True) and 1/128.0 normalization."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 2] = 255  # pure red in BGR

    blob = scrfd_model.preprocess(img)

    assert blob.shape == (1, 3, 64, 64)
    assert blob.dtype == np.float32
    assert np.allclose(blob[0, 0], (255.0 - 127.5) / 128.0), 'Channel 0 should be red'
    assert np.allclose(blob[0, 1], -127.5 / 128.0)
    assert np.allclose(blob[0, 2], -127.5 / 128.0), 'Channel 2 should be blue'


@pytest.mark.parametrize('input_size', [(640, 360), (1280, 720), (360, 640)])
def test_scrfd_non_multiple_of_32_input_size(input_size):
    """16:9 sizes like 640x360 and 1280x720 (multiples of 8, not 32) run without shape mismatch."""
    from pathlib import Path

    import cv2

    img_path = Path(__file__).resolve().parent.parent / 'assets' / 'source' / 'detect_group.jpg'
    image = cv2.imread(str(img_path))
    assert image is not None

    detector = SCRFD(model_name=SCRFDWeights.SCRFD_500M_KPS, input_size=input_size, providers=['CPUExecutionProvider'])
    faces = detector.detect(image)

    assert len(faces) > 0
    for face in faces:
        assert face.bbox.dtype == np.float32
        assert face.landmarks.dtype == np.float32


@pytest.mark.parametrize('bad_size', [(0, 640), (-64, 640), (641, 361)])
def test_scrfd_rejects_invalid_input_size(bad_size):
    with pytest.raises(ValueError, match='multiples of 8'):
        SCRFD(input_size=bad_size)


@pytest.mark.parametrize('bad_image', [None, np.array([1, 2, 3], dtype=np.uint8)])
def test_scrfd_validates_image_before_unpacking_shape(scrfd_model, bad_image):
    with pytest.raises(ValueError):
        scrfd_model.detect(bad_image)
