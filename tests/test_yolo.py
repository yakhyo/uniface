# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from uniface.common import letterbox_resize, resize_image
from uniface.detection import YOLOv5Face, YOLOv8Face

TEST_IMAGE = Path(__file__).resolve().parent.parent / 'assets' / 'source' / 'detect_group.jpg'


@pytest.fixture(scope='module')
def yolov5_model():
    return YOLOv5Face(providers=['CPUExecutionProvider'])


@pytest.fixture(scope='module')
def yolov8_model():
    return YOLOv8Face(providers=['CPUExecutionProvider'])


def test_yolov8_returns_float32_arrays(yolov8_model):
    image = cv2.imread(str(TEST_IMAGE))
    assert image is not None

    faces = yolov8_model.detect(image)
    assert len(faces) > 0
    for face in faces:
        assert face.bbox.dtype == np.float32
        assert face.landmarks.dtype == np.float32


def test_yolov5_returns_float32_arrays(yolov5_model):
    image = cv2.imread(str(TEST_IMAGE))
    assert image is not None

    faces = yolov5_model.detect(image)
    assert len(faces) > 0
    for face in faces:
        assert face.bbox.dtype == np.float32
        assert face.landmarks.dtype == np.float32


def test_yolo_empty_postprocess_shapes(yolov5_model, yolov8_model):
    empty_v5 = np.zeros((1, 25200, 16), dtype=np.float32)
    dets5, lms5 = yolov5_model.postprocess(empty_v5, 1.0, (0, 0))
    assert dets5.shape == (0, 5) and dets5.dtype == np.float32
    assert lms5.shape == (0, 5, 2) and lms5.dtype == np.float32

    empty_v8 = [np.full((1, 80, h, h), -1000.0, dtype=np.float32) for h in (80, 40, 20)]
    with np.errstate(over='raise'):
        dets8, lms8 = yolov8_model.postprocess(empty_v8, 1.0, (0, 0), (640, 640))
    assert dets8.shape == (0, 5) and dets8.dtype == np.float32
    assert lms8.shape == (0, 5, 2) and lms8.dtype == np.float32


@pytest.mark.parametrize('bad_image', [None, np.array([1, 2, 3], dtype=np.uint8)])
def test_yolo_validates_image_before_unpacking_shape(yolov5_model, yolov8_model, bad_image):
    with pytest.raises(ValueError):
        yolov5_model.detect(bad_image)
    with pytest.raises(ValueError):
        yolov8_model.detect(bad_image)


@pytest.mark.parametrize('shape', [(200, 1, 3), (1, 200, 3), (1000, 1, 3), (1, 1000, 3)])
def test_resize_helpers_handle_extreme_aspect_ratios(shape):
    thin = np.zeros(shape, dtype=np.uint8)

    canvas, factor = resize_image(thin, target_shape=(640, 640))
    assert canvas.shape == (640, 640, 3)
    assert factor == pytest.approx(min(640 / shape[0], 640 / shape[1]))

    blob, scale, _ = letterbox_resize(thin, target_size=128)
    assert blob.shape == (1, 3, 128, 128)
    assert scale > 0
