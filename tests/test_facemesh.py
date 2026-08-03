# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


from __future__ import annotations

import numpy as np
import pytest

from uniface.constants import FaceMeshWeights
from uniface.landmark import IRIS_LEFT, IRIS_RIGHT, NUM_MESH_LANDMARKS, FaceMesh, roi_from_box
from uniface.types import FaceMeshResult


@pytest.fixture(scope='module', params=list(FaceMeshWeights), ids=lambda w: w.name)
def mesher(request):
    """Every shared test runs against both generations.

    Assertions here go through `num_landmarks` rather than a literal, so the suite
    covers 468 and 478 with one body and a third model would need no new tests.
    """
    return FaceMesh(model_name=request.param)


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


@pytest.mark.parametrize(
    ('weights', 'num_landmarks', 'input_size'),
    [(FaceMeshWeights.V1_468, 468, 192), (FaceMeshWeights.V2_478, 478, 256)],
)
def test_model_geometry(weights, num_landmarks, input_size):
    """Both are read from the ONNX graph, never hardcoded in the class."""
    mesher = FaceMesh(model_name=weights)

    assert mesher.num_landmarks == num_landmarks
    assert mesher.input_size == input_size


def test_get_landmarks_shape(mesher, mock_image, mock_bbox):
    """The BaseLandmarker contract: 2D only, one face."""
    landmarks = mesher.get_landmarks(mock_image, mock_bbox)

    assert landmarks.shape == (mesher.num_landmarks, 2)
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
    assert results[0].landmarks.shape == (mesher.num_landmarks, 3)
    assert results[0].landmarks.dtype == np.float32
    assert 0.0 <= results[0].score <= 1.0
    assert results[0].points_2d.shape == (mesher.num_landmarks, 2)


def test_predict_batches_multiple_faces(mesher, mock_image, mock_bbox):
    """All faces go through a single batched session call."""
    boxes = [mock_bbox, mock_bbox + 50, mock_bbox + 100]
    results = mesher.predict(mock_image, bboxes=boxes)

    assert len(results) == 3
    assert all(r.landmarks.shape == (mesher.num_landmarks, 3) for r in results)


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
    assert all(r.landmarks.shape == (mesher.num_landmarks, 3) for r in results)


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


# Iris landmarks — indexed through the shared constants, MediaPipe's own convention
def test_iris_slices_cover_the_extra_points(mesher, mock_image, mock_bbox):
    """The 478-point model appends exactly the two irises after the mesh."""
    result = mesher.predict(mock_image, bboxes=[mock_bbox])[0]

    if mesher.num_landmarks == NUM_MESH_LANDMARKS:
        assert result.landmarks[IRIS_LEFT].shape == (0, 3)  # nothing past the mesh
        return

    assert result.landmarks[IRIS_LEFT].shape == (5, 3)
    assert result.landmarks[IRIS_RIGHT].shape == (5, 3)
    assert len(result.landmarks) == NUM_MESH_LANDMARKS + 10


@pytest.mark.parametrize(
    ('label', 'bad_image'),
    [
        ('float [0, 1]', np.random.rand(640, 640, 3).astype(np.float32)),
        ('grayscale', np.zeros((640, 640), dtype=np.uint8)),
        ('empty', np.zeros((0, 0, 3), dtype=np.uint8)),
    ],
)
def test_predict_rejects_unusable_images(mesher, mock_bbox, label, bad_image):
    """`preprocess` divides by 255, so a float image would silently mesh garbage."""
    with pytest.raises(ValueError):
        mesher.predict(bad_image, bboxes=[mock_bbox])


def test_get_landmarks_rejects_unusable_images(mesher, mock_bbox):
    """The score that would flag the bad input is dropped on this path."""
    with pytest.raises(ValueError, match='uint8'):
        mesher.get_landmarks(np.random.rand(640, 640, 3).astype(np.float32), mock_bbox)


def test_iris_constants_are_contiguous_and_ordered():
    """468-472 then 473-477, with no gap and no overlap."""
    assert (IRIS_LEFT.start, IRIS_LEFT.stop) == (NUM_MESH_LANDMARKS, NUM_MESH_LANDMARKS + 5)
    assert (IRIS_RIGHT.start, IRIS_RIGHT.stop) == (NUM_MESH_LANDMARKS + 5, NUM_MESH_LANDMARKS + 10)


def test_irises_sit_inside_the_face(mesher, mock_image, mock_bbox, mock_keypoints):
    """Catches an off-by-one in the slice or a bad inverse transform."""
    if mesher.num_landmarks == NUM_MESH_LANDMARKS:
        pytest.skip('468-point model has no irises')

    result = mesher.predict(mock_image, bboxes=[mock_bbox], keypoints=[mock_keypoints])[0]
    mesh = result.landmarks[:NUM_MESH_LANDMARKS, :2]

    for iris in (result.landmarks[IRIS_LEFT], result.landmarks[IRIS_RIGHT]):
        assert np.all(iris[:, 0] > mesh[:, 0].min()) and np.all(iris[:, 0] < mesh[:, 0].max())
        assert np.all(iris[:, 1] > mesh[:, 1].min()) and np.all(iris[:, 1] < mesh[:, 1].max())
