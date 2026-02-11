# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    import lap

    LAP_AVAILABLE = True
except ImportError:
    LAP_AVAILABLE = False


def linear_assignment(
    cost_matrix: np.ndarray, thresh: float, use_lap: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform linear assignment using lap or scipy.

    Args:
        cost_matrix: Cost matrix of shape (N, M).
        thresh: Maximum cost threshold for valid assignment.
        use_lap: Use lap.lapjv (faster) if available.

    Returns:
        Tuple of (matches, unmatched_a, unmatched_b).
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0]),
            np.arange(cost_matrix.shape[1]),
        )

    if use_lap and LAP_AVAILABLE:
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
    else:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = [[r, c] for r, c in zip(row_ind, col_ind, strict=False) if cost_matrix[r, c] <= thresh]
        if len(matches) == 0:
            unmatched_a = np.arange(cost_matrix.shape[0])
            unmatched_b = np.arange(cost_matrix.shape[1])
        else:
            matches_arr = np.array(matches)
            unmatched_a = np.array([i for i in range(cost_matrix.shape[0]) if i not in matches_arr[:, 0]])
            unmatched_b = np.array([i for i in range(cost_matrix.shape[1]) if i not in matches_arr[:, 1]])

    matches = np.asarray(matches) if matches else np.empty((0, 2), dtype=int)
    return matches, unmatched_a, unmatched_b


def iou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of bounding boxes.

    Args:
        bboxes1: Array of shape (N, 4) in [x1, y1, x2, y2] format.
        bboxes2: Array of shape (M, 4) in [x1, y1, x2, y2] format.

    Returns:
        IoU matrix of shape (N, M).
    """
    bboxes1 = np.ascontiguousarray(bboxes1, dtype=np.float32)
    bboxes2 = np.ascontiguousarray(bboxes2, dtype=np.float32)

    rows, cols = bboxes1.shape[0], bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)

    if rows * cols == 0:
        return ious

    x1 = np.maximum(bboxes1[:, None, 0], bboxes2[None, :, 0])
    y1 = np.maximum(bboxes1[:, None, 1], bboxes2[None, :, 1])
    x2 = np.minimum(bboxes1[:, None, 2], bboxes2[None, :, 2])
    y2 = np.minimum(bboxes1[:, None, 3], bboxes2[None, :, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    union = area1[:, None] + area2[None, :] - intersection
    ious = intersection / np.maximum(union, 1e-10)

    return ious


def iou_distance(atracks: list, btracks: list) -> np.ndarray:
    """Compute cost matrix based on IoU distance.

    Args:
        atracks: List of tracks or bounding boxes.
        btracks: List of tracks or bounding boxes.

    Returns:
        Cost matrix (1 - IoU) of shape (len(atracks), len(btracks)).
    """
    if len(atracks) > 0 and isinstance(atracks[0], np.ndarray):
        atlbrs = atracks
    else:
        atlbrs = [track.tlbr for track in atracks]

    if len(btracks) > 0 and isinstance(btracks[0], np.ndarray):
        btlbrs = btracks
    else:
        btlbrs = [track.tlbr for track in btracks]

    atlbrs = np.asarray(atlbrs) if len(atlbrs) > 0 else np.empty((0, 4))
    btlbrs = np.asarray(btlbrs) if len(btlbrs) > 0 else np.empty((0, 4))

    ious = iou_batch(atlbrs, btlbrs)
    return 1 - ious
