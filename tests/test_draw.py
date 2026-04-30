# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import numpy as np

from uniface.draw import FACE_PARSING_COLORS, draw_gaze, vis_parsing_maps


def _vis_parsing_maps_reference(image: np.ndarray, segmentation_mask: np.ndarray) -> np.ndarray:
    """Original per-class np.where implementation, used as a regression baseline."""
    import cv2

    image = np.array(image).copy().astype(np.uint8)
    segmentation_mask = segmentation_mask.copy().astype(np.uint8)
    color = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1], 3))
    num_classes = np.max(segmentation_mask)
    for class_index in range(1, num_classes + 1):
        pixels = np.where(segmentation_mask == class_index)
        color[pixels[0], pixels[1], :] = FACE_PARSING_COLORS[class_index]
    color = color.astype(np.uint8)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(bgr, 0.6, color, 0.4, 0)


def test_vis_parsing_maps_matches_reference_random_mask():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    mask = rng.integers(0, len(FACE_PARSING_COLORS), size=(128, 128), dtype=np.uint8)

    expected = _vis_parsing_maps_reference(image, mask)
    actual = vis_parsing_maps(image, mask)

    assert actual.dtype == np.uint8
    assert actual.shape == (128, 128, 3)
    assert np.array_equal(actual, expected)


def test_vis_parsing_maps_all_zero_mask_is_dimmed_image():
    image = np.full((32, 32, 3), 200, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)

    expected = _vis_parsing_maps_reference(image, mask)
    actual = vis_parsing_maps(image, mask)
    assert np.array_equal(actual, expected)


def test_vis_parsing_maps_full_class_range():
    """Mask covering every defined class index — checks palette wiring end-to-end."""
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    mask = np.tile(np.arange(len(FACE_PARSING_COLORS), dtype=np.uint8), 64 * 64 // len(FACE_PARSING_COLORS) + 1)[
        : 64 * 64
    ].reshape(64, 64)

    expected = _vis_parsing_maps_reference(image, mask)
    actual = vis_parsing_maps(image, mask)
    assert np.array_equal(actual, expected)


def _compute_gaze_delta(bbox: np.ndarray, pitch: float, yaw: float) -> tuple[int, int]:
    """Replicate draw_gaze dx/dy math for verification."""
    x_min, _, x_max, _ = map(int, bbox[:4])
    length = x_max - x_min
    dx = int(-length * np.sin(yaw) * np.cos(pitch))
    dy = int(-length * np.sin(pitch))
    return dx, dy


def test_draw_gaze_yaw_only_moves_horizontally():
    """Yaw-only input (pitch=0) should produce horizontal displacement only."""
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    bbox = np.array([50, 50, 150, 150], dtype=np.float32)

    yaw = 0.5
    pitch = 0.0
    dx, dy = _compute_gaze_delta(bbox, pitch, yaw)

    assert dx != 0, 'Yaw-only should produce horizontal displacement'
    assert dy == 0, 'Yaw-only should produce zero vertical displacement'

    # Should not raise
    draw_gaze(image, bbox, pitch, yaw, draw_bbox=False, draw_angles=False)


def test_draw_gaze_pitch_only_moves_vertically():
    """Pitch-only input (yaw=0) should produce vertical displacement only."""
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    bbox = np.array([50, 50, 150, 150], dtype=np.float32)

    yaw = 0.0
    pitch = 0.5
    dx, dy = _compute_gaze_delta(bbox, pitch, yaw)

    assert dx == 0, 'Pitch-only should produce zero horizontal displacement'
    assert dy != 0, 'Pitch-only should produce vertical displacement'

    # Should not raise
    draw_gaze(image, bbox, pitch, yaw, draw_bbox=False, draw_angles=False)


def test_draw_gaze_modifies_image():
    """draw_gaze should modify the image in place."""
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    bbox = np.array([50, 50, 150, 150], dtype=np.float32)

    original = image.copy()
    draw_gaze(image, bbox, 0.3, 0.3)

    assert not np.array_equal(image, original), 'draw_gaze should modify the image'
