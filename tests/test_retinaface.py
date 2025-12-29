# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Tests for RetinaFace detector."""

from __future__ import annotations

import numpy as np
import pytest

from uniface.constants import RetinaFaceWeights
from uniface.detection import RetinaFace


@pytest.fixture
def retinaface_model():
    return RetinaFace(
        model_name=RetinaFaceWeights.MNET_V2,
        confidence_threshold=0.5,
        pre_nms_topk=5000,
        nms_threshold=0.4,
        post_nms_topk=750,
    )


def test_model_initialization(retinaface_model):
    assert retinaface_model is not None, 'Model initialization failed.'


def test_inference_on_640x640_image(retinaface_model):
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    faces = retinaface_model.detect(mock_image)

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


def test_confidence_threshold(retinaface_model):
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    faces = retinaface_model.detect(mock_image)

    for face in faces:
        confidence = face.confidence
        assert confidence >= 0.5, f'Detection has confidence {confidence} below threshold 0.5'


def test_no_faces_detected(retinaface_model):
    empty_image = np.zeros((640, 640, 3), dtype=np.uint8)
    faces = retinaface_model.detect(empty_image)
    assert len(faces) == 0, 'Should detect no faces in a blank image.'
