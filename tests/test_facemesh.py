# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


from __future__ import annotations

import numpy as np
import pytest

from uniface.landmark import FaceMesh, roi_from_box
from uniface.types import FaceMeshResult


@pytest.fixture(scope='module')
def mesher():
    return FaceMesh()


@pytest.fixture
def mock_image():
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_bbox():
    return np.array([100, 100, 300, 300], dtype=np.float32)


@pytest.fixture
def mock_keypoints():
    """Five alignment landmarks with a horizontal eye line."""
    return np.array(
        [[150.0, 160.0], [250.0, 160.0], [200.0, 210.0], [160.0, 260.0], [240.0, 260.0]],
        dtype=np.float32,
    )


def test_model_initialization(mesher):
    assert mesher.num_landmarks == 468
    assert mesher.input_size == 192


def test_get_landmarks_shape(mesher, mock_image, mock_bbox):
    """The BaseLandmarker contract: 2D only, one face."""
    landmarks = mesher.get_landmarks(mock_image, mock_bbox)

    assert landmarks.shape == (468, 2)
    assert landmarks.dtype == np.float32


def test_call_shortcut_forwards_keypoints(mesher, mock_image, mock_bbox, mock_keypoints):
    """__call__ is overridden; the base class signature would drop keypoints."""
    direct = mesher.get_landmarks(mock_image, mock_bbox, mock_keypoints)
    called = mesher(mock_image, mock_bbox, mock_keypoints)

    assert np.allclose(direct, called)


def test_predict_returns_results(mesher, mock_image, mock_bbox):
    results = mesher.predict(mock_image, bboxes=[mock_bbox])

    assert len(results) == 1
    assert isinstance(results[0], FaceMeshResult)
    assert results[0].landmarks.shape == (468, 3)
    assert results[0].landmarks.dtype == np.float32
    assert 0.0 <= results[0].score <= 1.0
    assert results[0].points_2d.shape == (468, 2)


def test_predict_batches_multiple_faces(mesher, mock_image, mock_bbox):
    """All faces go through a single batched session call."""
    boxes = [mock_bbox, mock_bbox + 50, mock_bbox + 100]
    results = mesher.predict(mock_image, bboxes=boxes)

    assert len(results) == 3
    assert all(r.landmarks.shape == (468, 3) for r in results)


def test_predict_input_validation(mesher, mock_image, mock_bbox, mock_keypoints):
    """faces and bboxes are mutually exclusive, and keypoints must line up."""
    assert mesher.predict(mock_image, bboxes=[]) == []

    with pytest.raises(ValueError, match='either faces or bboxes'):
        mesher.predict(mock_image)

    with pytest.raises(ValueError, match='either faces or bboxes'):
        mesher.predict(mock_image, [], bboxes=[mock_bbox])

    with pytest.raises(ValueError, match='keypoint sets'):
        mesher.predict(mock_image, bboxes=[mock_bbox, mock_bbox], keypoints=[mock_keypoints])


def test_depth_is_populated(mesher, mock_image, mock_bbox, mock_keypoints):
    """z must carry real depth, not a zero-filled placeholder."""
    result = mesher.predict(mock_image, bboxes=[mock_bbox], keypoints=[mock_keypoints])[0]

    assert not np.allclose(result.landmarks[:, 2], 0.0)
    assert np.allclose(result.points_2d, result.landmarks[:, :2])


def test_landmarks_map_back_to_the_image(mesher, mock_image, mock_bbox, mock_keypoints):
    """Catches an inverted or mis-scaled ROI transform."""
    result = mesher.predict(mock_image, bboxes=[mock_bbox], keypoints=[mock_keypoints])[0]
    points = result.landmarks[:, :2]

    assert 100 <= points[:, 0].mean() <= 300
    assert 100 <= points[:, 1].mean() <= 300


def test_roll_normalization_changes_the_result(mesher, mock_image, mock_bbox, mock_keypoints):
    """Rotating the eye line must rotate the ROI, and so change the landmarks."""
    tilted = mock_keypoints.copy()
    tilted[1] = [230.0, 240.0]  # drop the right eye to tilt the eye line

    upright = mesher.get_landmarks(mock_image, mock_bbox, mock_keypoints)
    rotated = mesher.get_landmarks(mock_image, mock_bbox, tilted)

    assert not np.allclose(upright, rotated)


def test_works_with_any_detector(mesher, mock_image):
    """The cross-detector guarantee: Face objects from SCRFD feed FaceMesh directly."""
    from uniface.detection import SCRFD

    faces = SCRFD().detect(mock_image)
    results = mesher.predict(mock_image, faces)

    assert len(results) == len(faces)
    assert all(r.landmarks.shape == (468, 3) for r in results)


# roi_from_box — MediaPipe's detection_to_roi rule
def test_roi_is_square_on_the_long_side(mock_bbox):
    center_x, center_y, side, angle = roi_from_box(mock_bbox, margin=0.25)

    assert (center_x, center_y) == (200.0, 200.0)
    assert side == pytest.approx(200.0 * 1.5)  # MediaPipe's 1.5x scale
    assert angle == 0.0

    # Never a stretched rectangle: the long side wins.
    assert roi_from_box(np.array([0, 0, 100, 200]), margin=0.0)[2] == pytest.approx(200.0)


def test_roi_angle_comes_from_the_eyes(mock_bbox):
    level = np.array([[0.0, 0.0], [10.0, 0.0]])
    tilted = np.array([[0.0, 0.0], [10.0, 10.0]])
    # BlazeFace's 6-point layout also has the eyes in rows 0/1.
    six = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 5.0], [5.0, 8.0], [-2.0, 2.0], [12.0, 2.0]])

    assert roi_from_box(mock_bbox, level)[3] == pytest.approx(0.0)
    assert roi_from_box(mock_bbox, tilted)[3] == pytest.approx(45.0)
    assert roi_from_box(mock_bbox, six)[3] == pytest.approx(0.0)


def test_result_equality_does_not_raise():
    """eq=False: comparison falls back to identity instead of an ndarray ValueError."""
    a = FaceMeshResult(landmarks=np.zeros((468, 3), dtype=np.float32), score=1.0)
    b = FaceMeshResult(landmarks=np.zeros((468, 3), dtype=np.float32), score=1.0)

    assert a == a
    assert a != b  # identity, not value semantics
    assert hash(a) is not None
