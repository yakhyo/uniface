# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Tests for SCRFD detector."""

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
