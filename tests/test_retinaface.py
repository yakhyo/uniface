# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from uniface.constants import RetinaFaceWeights
from uniface.detection import RetinaFace

TEST_IMAGE = Path(__file__).resolve().parent.parent / 'assets' / 'source' / 'detect_group.jpg'


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


@pytest.mark.parametrize('dynamic_size', [False, True])
@pytest.mark.parametrize(
    'bad_image',
    [
        np.zeros((240, 320, 3), dtype=np.float32),
        np.zeros((240, 320), dtype=np.uint8),
        np.zeros((240, 320, 4), dtype=np.uint8),
        np.zeros((0, 0, 3), dtype=np.uint8),
    ],
    ids=['float', 'grayscale', 'bgra', 'empty'],
)
def test_invalid_image_is_rejected(bad_image, dynamic_size):
    """Both sizing modes reject an image the model cannot read instead of returning no faces."""
    detector = RetinaFace(dynamic_size=dynamic_size, providers=['CPUExecutionProvider'])

    with pytest.raises(ValueError, match=r'uint8|BGR image|non-empty'):
        detector.detect(bad_image)


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return float(intersection / (area_a + area_b - intersection))


@pytest.mark.parametrize('input_size', [(640, 480), (480, 640)])
def test_non_square_input_size_matches_square(input_size):
    """A non-square (width, height) input size finds the same faces as the default square size."""
    image = cv2.imread(str(TEST_IMAGE))
    assert image is not None, f'Could not read {TEST_IMAGE}'

    providers = ['CPUExecutionProvider']
    reference = RetinaFace(providers=providers).detect(image)
    faces = RetinaFace(input_size=input_size, providers=providers).detect(image)

    assert len(reference) > 0, 'Reference detector should find faces in the group photo.'
    assert len(faces) >= len(reference) - 1, f'Found {len(faces)} faces, expected about {len(reference)}.'

    # Misplaced anchors produce confident boxes that overlap no real face
    for face in faces:
        best_iou = max(_iou(face.bbox, ref.bbox) for ref in reference)
        assert best_iou >= 0.5, f'Box {face.bbox} matches no face found at the default size.'
