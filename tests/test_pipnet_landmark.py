# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


from __future__ import annotations

import numpy as np
import pytest

from uniface.constants import PIPNetWeights
from uniface.landmark import PIPNet


@pytest.fixture(scope='module', params=[PIPNetWeights.WFLW_98, PIPNetWeights.DW300_CELEBA_68])
def pipnet_model(request):
    return PIPNet(model_name=request.param)


@pytest.fixture
def mock_image():
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_bbox():
    return [100, 100, 300, 300]


def _expected_n_lms(model: PIPNet) -> int:
    return 98 if model.num_lms == 98 else 68


def test_model_initialization(pipnet_model):
    assert pipnet_model is not None, 'PIPNet model initialization failed.'
    assert pipnet_model.num_lms in (68, 98), f'Unexpected num_lms: {pipnet_model.num_lms}'
    assert pipnet_model.input_h == pipnet_model.input_w == 256


def test_landmark_detection(pipnet_model, mock_image, mock_bbox):
    landmarks = pipnet_model.get_landmarks(mock_image, mock_bbox)
    n = _expected_n_lms(pipnet_model)
    assert landmarks.shape == (n, 2), f'Expected shape ({n}, 2), got {landmarks.shape}'


def test_landmark_dtype(pipnet_model, mock_image, mock_bbox):
    landmarks = pipnet_model.get_landmarks(mock_image, mock_bbox)
    assert landmarks.dtype == np.float32, f'Expected float32, got {landmarks.dtype}'


def test_landmark_coordinates_within_image(pipnet_model, mock_image, mock_bbox):
    landmarks = pipnet_model.get_landmarks(mock_image, mock_bbox)
    n = _expected_n_lms(pipnet_model)

    x_coords = landmarks[:, 0]
    y_coords = landmarks[:, 1]

    x1, y1, x2, y2 = mock_bbox
    margin = 50

    x_in_bounds = int(np.sum((x_coords >= x1 - margin) & (x_coords <= x2 + margin)))
    y_in_bounds = int(np.sum((y_coords >= y1 - margin) & (y_coords <= y2 + margin)))

    threshold = max(int(0.9 * n), n - 5)
    assert x_in_bounds >= threshold, f'Only {x_in_bounds}/{n} x-coordinates within bounds'
    assert y_in_bounds >= threshold, f'Only {y_in_bounds}/{n} y-coordinates within bounds'


def test_different_bbox_sizes(pipnet_model, mock_image):
    n = _expected_n_lms(pipnet_model)
    test_bboxes = [
        [50, 50, 150, 150],
        [100, 100, 300, 300],
        [50, 50, 400, 400],
    ]

    for bbox in test_bboxes:
        landmarks = pipnet_model.get_landmarks(mock_image, bbox)
        assert landmarks.shape == (n, 2), f'Failed for bbox {bbox}'


def test_consistency(pipnet_model, mock_image, mock_bbox):
    landmarks1 = pipnet_model.get_landmarks(mock_image, mock_bbox)
    landmarks2 = pipnet_model.get_landmarks(mock_image, mock_bbox)
    assert np.allclose(landmarks1, landmarks2), 'Same input should produce same landmarks'


def test_different_image_sizes(pipnet_model, mock_bbox):
    n = _expected_n_lms(pipnet_model)
    test_sizes = [(480, 640, 3), (720, 1280, 3), (1080, 1920, 3)]

    for size in test_sizes:
        mock_image = np.random.randint(0, 255, size, dtype=np.uint8)
        landmarks = pipnet_model.get_landmarks(mock_image, mock_bbox)
        assert landmarks.shape == (n, 2), f'Failed for image size {size}'


def test_bbox_list_format(pipnet_model, mock_image):
    n = _expected_n_lms(pipnet_model)
    landmarks = pipnet_model.get_landmarks(mock_image, [100, 100, 300, 300])
    assert landmarks.shape == (n, 2), 'Should work with bbox as list'


def test_bbox_array_format(pipnet_model, mock_image):
    n = _expected_n_lms(pipnet_model)
    bbox_array = np.array([100, 100, 300, 300])
    landmarks = pipnet_model.get_landmarks(mock_image, bbox_array)
    assert landmarks.shape == (n, 2), 'Should work with bbox as numpy array'


def test_landmark_distribution(pipnet_model, mock_image, mock_bbox):
    landmarks = pipnet_model.get_landmarks(mock_image, mock_bbox)

    x_variance = np.var(landmarks[:, 0])
    y_variance = np.var(landmarks[:, 1])

    assert x_variance > 0, 'Landmarks should have variation in x-coordinates'
    assert y_variance > 0, 'Landmarks should have variation in y-coordinates'


def test_default_model_is_wflw_98():
    """PIPNet() with no args should default to the 98-point WFLW model."""
    model = PIPNet()
    assert model.num_lms == 98


def test_meanface_lookup_invalid_num_lms():
    """get_meanface_info should reject unsupported landmark counts."""
    from uniface.landmark._meanface import get_meanface_info

    with pytest.raises(ValueError, match='No meanface table'):
        get_meanface_info(num_lms=42)
