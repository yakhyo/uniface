# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import itertools
import math

import cv2
import numpy as np

__all__ = [
    'decode_boxes',
    'decode_landmarks',
    'distance2bbox',
    'distance2kps',
    'generate_anchors',
    'non_max_suppression',
    'resize_image',
]


def resize_image(
    frame: np.ndarray,
    target_shape: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, float]:
    """Resize an image to fit within a target shape while keeping its aspect ratio.

    The image is resized to fit within the target dimensions and placed on a
    blank canvas (zero-padded to target size).

    Args:
        frame: Input image with shape (H, W, C).
        target_shape: Target size as (width, height). Defaults to (640, 640).

    Returns:
        A tuple containing:
            - Resized image on a blank canvas with shape (height, width, 3).
            - The resize factor as a float.
    """
    width, height = target_shape

    # Aspect-ratio preserving resize
    im_ratio = float(frame.shape[0]) / frame.shape[1]
    model_ratio = height / width
    if im_ratio > model_ratio:
        new_height = height
        new_width = int(new_height / im_ratio)
    else:
        new_width = width
        new_height = int(new_width * im_ratio)

    resize_factor = float(new_height) / frame.shape[0]
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Create blank image and place resized image on it
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:new_height, :new_width, :] = resized_frame

    return image, resize_factor


def generate_anchors(image_size: tuple[int, int] = (640, 640)) -> np.ndarray:
    """Generate anchor boxes for a given image size (RetinaFace specific).

    Args:
        image_size: Input image size as (width, height). Defaults to (640, 640).

    Returns:
        Anchor box coordinates as a numpy array with shape (num_anchors, 4).
    """
    # RetinaFace FPN strides and corresponding anchor sizes per level
    steps = [8, 16, 32]
    min_sizes = [[16, 32], [64, 128], [256, 512]]

    anchors = []
    feature_maps = [[math.ceil(image_size[0] / step), math.ceil(image_size[1] / step)] for step in steps]

    for k, (map_height, map_width) in enumerate(feature_maps):
        step = steps[k]
        for i, j in itertools.product(range(map_height), range(map_width)):
            for min_size in min_sizes[k]:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]

                dense_cx = [x * step / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * step / image_size[0] for y in [i + 0.5]]
                for cy, cx in itertools.product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]

    output = np.array(anchors, dtype=np.float32).reshape(-1, 4)
    return output


def non_max_suppression(dets: np.ndarray, threshold: float) -> list[int]:
    """Apply Non-Maximum Suppression (NMS) to reduce overlapping bounding boxes.

    Args:
        dets: Array of detections with each row as [x1, y1, x2, y2, score].
        threshold: IoU threshold for suppression.

    Returns:
        Indices of bounding boxes retained after suppression.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep


def decode_boxes(
    loc: np.ndarray,
    priors: np.ndarray,
    variances: list[float] | None = None,
) -> np.ndarray:
    """Decode locations from predictions using priors (RetinaFace specific).

    Undoes the encoding done for offset regression at train time.

    Args:
        loc: Location predictions for loc layers, shape: [num_priors, 4].
        priors: Prior boxes in center-offset form, shape: [num_priors, 4].
        variances: Variances of prior boxes. Defaults to [0.1, 0.2].

    Returns:
        Decoded bounding box predictions with shape [num_priors, 4].
    """
    if variances is None:
        variances = [0.1, 0.2]
    # Compute centers of predicted boxes
    cxcy = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]

    # Compute widths and heights of predicted boxes
    wh = priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])

    # Convert center, size to corner coordinates
    boxes = np.zeros_like(loc)
    boxes[:, :2] = cxcy - wh / 2  # xmin, ymin
    boxes[:, 2:] = cxcy + wh / 2  # xmax, ymax

    return boxes


def decode_landmarks(
    predictions: np.ndarray,
    priors: np.ndarray,
    variances: list[float] | None = None,
) -> np.ndarray:
    """Decode landmark predictions using prior boxes (RetinaFace specific).

    Args:
        predictions: Landmark predictions, shape: [num_priors, 10].
        priors: Prior boxes, shape: [num_priors, 4].
        variances: Scaling factors for landmark offsets. Defaults to [0.1, 0.2].

    Returns:
        Decoded landmarks, shape: [num_priors, 10].
    """
    if variances is None:
        variances = [0.1, 0.2]

    # Reshape predictions to [num_priors, 5, 2] to process landmark points
    predictions = predictions.reshape(predictions.shape[0], 5, 2)

    # Expand priors to match (num_priors, 5, 2)
    priors_xy = np.repeat(priors[:, :2][:, np.newaxis, :], 5, axis=1)  # (num_priors, 5, 2)
    priors_wh = np.repeat(priors[:, 2:][:, np.newaxis, :], 5, axis=1)  # (num_priors, 5, 2)

    # Compute absolute landmark positions
    landmarks = priors_xy + predictions * variances[0] * priors_wh

    # Flatten back to [num_priors, 10]
    landmarks = landmarks.reshape(landmarks.shape[0], -1)

    return landmarks


def distance2bbox(
    points: np.ndarray,
    distance: np.ndarray,
    max_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Decode distance prediction to bounding box (SCRFD specific).

    Args:
        points: Anchor points with shape (n, 2), [x, y].
        distance: Distance from the given point to 4 boundaries
            (left, top, right, bottom) with shape (n, 4).
        max_shape: Shape of the image (height, width) for clipping.

    Returns:
        Decoded bounding boxes with shape (n, 4) as [x1, y1, x2, y2].
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]

    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    else:
        x1 = np.maximum(x1, 0)
        y1 = np.maximum(y1, 0)
        x2 = np.maximum(x2, 0)
        y2 = np.maximum(y2, 0)

    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(
    points: np.ndarray,
    distance: np.ndarray,
    max_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Decode distance prediction to keypoints (SCRFD specific).

    Args:
        points: Anchor points with shape (n, 2), [x, y].
        distance: Distance from the given point to keypoints with shape (n, 2k).
        max_shape: Shape of the image (height, width) for clipping.

    Returns:
        Decoded keypoints with shape (n, 2k).
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)
