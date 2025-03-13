# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import cv2
import math
import itertools
import numpy as np

import torch
from typing import Tuple, List


def resize_image(frame, target_shape: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, float]:
    """
    Resize an image to fit within a target shape while keeping its aspect ratio.

    Args:
        frame (np.ndarray): Input image.
        target_shape (Tuple[int, int]): Target size (width, height). Defaults to (640, 640).

    Returns:
        Tuple[np.ndarray, float]: Resized image on a blank canvas and the resize factor.
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


def generate_anchors(image_size: Tuple[int, int] = (640, 640)) -> torch.Tensor:
    """
    Generate anchor boxes for a given image size.

    Args:
        image_size (Tuple[int, int]): Input image size (width, height). Defaults to (640, 640).

    Returns:
        torch.Tensor: Anchor box coordinates as a tensor.
    """
    image_size = image_size

    steps = [8, 16, 32]
    min_sizes = [[16, 32], [64, 128], [256, 512]]

    anchors = []
    feature_maps = [
        [
            math.ceil(image_size[0] / step),
            math.ceil(image_size[1] / step)
        ] for step in steps
    ]

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

    output = torch.Tensor(anchors).view(-1, 4)
    return output


def nms(dets: List[np.ndarray], threshold: float):
    """
    Apply Non-Maximum Suppression (NMS) to reduce overlapping bounding boxes based on a threshold.

    Args:
        dets (numpy.ndarray): Array of detections with each row as [x1, y1, x2, y2, score].
        threshold (float): IoU threshold for suppression.

    Returns:
        list: Indices of bounding boxes retained after suppression.
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


def decode_boxes(loc, priors, variances=[0.1, 0.2]) -> torch.Tensor:
    """
    Decode locations from predictions using priors to undo
    the encoding done for offset regression at train time.

    Args:
        loc (tensor): Location predictions for loc layers, shape: [num_priors, 4]
        priors (tensor): Prior boxes in center-offset form, shape: [num_priors, 4]
        variances (list[float]): Variances of prior boxes

    Returns:
        tensor: Decoded bounding box predictions
    """
    # Compute centers of predicted boxes
    cxcy = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]

    # Compute widths and heights of predicted boxes
    wh = priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])

    # Convert center, size to corner coordinates
    boxes = torch.empty_like(loc)
    boxes[:, :2] = cxcy - wh / 2  # xmin, ymin
    boxes[:, 2:] = cxcy + wh / 2  # xmax, ymax

    return boxes


def decode_landmarks(predictions, priors, variances=[0.1, 0.2]) -> torch.Tensor:
    """
    Decode landmarks from predictions using prior boxes to reverse the encoding done during training.

    Args:
        predictions (tensor): Landmark predictions for localization layers.
            Shape: [num_priors, 10] where each prior contains 5 landmark (x, y) pairs.
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors, 4], where each prior has (cx, cy, width, height).
        variances (list[float]): Variances of the prior boxes to scale the decoded values.

    Returns:
        landmarks (tensor): Decoded landmark predictions.
            Shape: [num_priors, 10] where each row contains the decoded (x, y) pairs for 5 landmarks.
    """

    # Reshape predictions to [num_priors, 5, 2] to handle each pair (x, y) in a batch
    predictions = predictions.view(predictions.size(0), 5, 2)

    # Perform the same operation on all landmark pairs at once
    landmarks = priors[:, :2].unsqueeze(1) + predictions * variances[0] * priors[:, 2:].unsqueeze(1)

    # Flatten back to [num_priors, 10]
    landmarks = landmarks.view(landmarks.size(0), -1)

    return landmarks
