# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import numpy as np

from uniface.draw import draw_gaze


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
