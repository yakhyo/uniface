# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Tests for 106-point facial landmark detector."""

from __future__ import annotations

import numpy as np
import pytest

from uniface.landmark import Landmark106


@pytest.fixture
def landmark_model():
    return Landmark106()


@pytest.fixture
def mock_image():
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_bbox():
    return [100, 100, 300, 300]


def test_model_initialization(landmark_model):
    assert landmark_model is not None, 'Landmark106 model initialization failed.'


def test_landmark_detection(landmark_model, mock_image, mock_bbox):
    landmarks = landmark_model.get_landmarks(mock_image, mock_bbox)
    assert landmarks.shape == (106, 2), f'Expected shape (106, 2), got {landmarks.shape}'


def test_landmark_dtype(landmark_model, mock_image, mock_bbox):
    landmarks = landmark_model.get_landmarks(mock_image, mock_bbox)
    assert landmarks.dtype == np.float32, f'Expected float32, got {landmarks.dtype}'


def test_landmark_coordinates_within_image(landmark_model, mock_image, mock_bbox):
    landmarks = landmark_model.get_landmarks(mock_image, mock_bbox)

    x_coords = landmarks[:, 0]
    y_coords = landmarks[:, 1]

    x1, y1, x2, y2 = mock_bbox
    margin = 50

    x_in_bounds = np.sum((x_coords >= x1 - margin) & (x_coords <= x2 + margin))
    y_in_bounds = np.sum((y_coords >= y1 - margin) & (y_coords <= y2 + margin))

    assert x_in_bounds >= 95, f'Only {x_in_bounds}/106 x-coordinates within bounds'
    assert y_in_bounds >= 95, f'Only {y_in_bounds}/106 y-coordinates within bounds'


def test_different_bbox_sizes(landmark_model, mock_image):
    test_bboxes = [
        [50, 50, 150, 150],
        [100, 100, 300, 300],
        [50, 50, 400, 400],
    ]

    for bbox in test_bboxes:
        landmarks = landmark_model.get_landmarks(mock_image, bbox)
        assert landmarks.shape == (106, 2), f'Failed for bbox {bbox}'


def test_landmark_array_format(landmark_model, mock_image, mock_bbox):
    landmarks = landmark_model.get_landmarks(mock_image, mock_bbox)
    landmarks_int = landmarks.astype(int)

    assert landmarks_int.shape == (106, 2), 'Integer conversion should preserve shape'
    assert landmarks_int.dtype in [np.int32, np.int64], 'Should convert to integer type'


def test_consistency(landmark_model, mock_image, mock_bbox):
    landmarks1 = landmark_model.get_landmarks(mock_image, mock_bbox)
    landmarks2 = landmark_model.get_landmarks(mock_image, mock_bbox)

    assert np.allclose(landmarks1, landmarks2), 'Same input should produce same landmarks'


def test_different_image_sizes(landmark_model, mock_bbox):
    test_sizes = [(480, 640, 3), (720, 1280, 3), (1080, 1920, 3)]

    for size in test_sizes:
        mock_image = np.random.randint(0, 255, size, dtype=np.uint8)
        landmarks = landmark_model.get_landmarks(mock_image, mock_bbox)
        assert landmarks.shape == (106, 2), f'Failed for image size {size}'


def test_bbox_list_format(landmark_model, mock_image):
    bbox_list = [100, 100, 300, 300]
    landmarks = landmark_model.get_landmarks(mock_image, bbox_list)
    assert landmarks.shape == (106, 2), 'Should work with bbox as list'


def test_bbox_array_format(landmark_model, mock_image):
    bbox_array = np.array([100, 100, 300, 300])
    landmarks = landmark_model.get_landmarks(mock_image, bbox_array)
    assert landmarks.shape == (106, 2), 'Should work with bbox as numpy array'


def test_landmark_distribution(landmark_model, mock_image, mock_bbox):
    landmarks = landmark_model.get_landmarks(mock_image, mock_bbox)

    x_variance = np.var(landmarks[:, 0])
    y_variance = np.var(landmarks[:, 1])

    assert x_variance > 0, 'Landmarks should have variation in x-coordinates'
    assert y_variance > 0, 'Landmarks should have variation in y-coordinates'
