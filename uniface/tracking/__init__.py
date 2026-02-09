# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import numpy as np

from .bytetrack import BYTETracker


def expand_bboxes(
    detections: np.ndarray,
    scale: float = 1.5,
    image_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Expand detection bounding boxes by a scale factor around their center.

    Useful for tracking small faces: expanding the bbox before feeding to the
    tracker improves IoU overlap between frames.

    Args:
        detections: Array of shape (N, 5) with [x1, y1, x2, y2, score].
        scale: Expansion factor (e.g. 1.5 = 50% larger). Defaults to 1.5.
        image_shape: (height, width) for clamping to image bounds.
            If None, no clamping is applied.

    Returns:
        Array of shape (N, 5) with expanded bboxes and original scores.
    """
    if len(detections) == 0:
        return detections.copy()

    expanded = detections.copy()
    bboxes = expanded[:, :4]

    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]

    new_w = w * scale
    new_h = h * scale

    expanded[:, 0] = cx - new_w / 2
    expanded[:, 1] = cy - new_h / 2
    expanded[:, 2] = cx + new_w / 2
    expanded[:, 3] = cy + new_h / 2

    if image_shape is not None:
        img_h, img_w = image_shape
        expanded[:, 0] = np.clip(expanded[:, 0], 0, img_w)
        expanded[:, 1] = np.clip(expanded[:, 1], 0, img_h)
        expanded[:, 2] = np.clip(expanded[:, 2], 0, img_w)
        expanded[:, 3] = np.clip(expanded[:, 3], 0, img_h)

    return expanded


__all__ = ['BYTETracker', 'expand_bboxes']
