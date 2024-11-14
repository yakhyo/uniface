import math
import numpy as np
from itertools import product

import torch
from typing import Tuple

import cv2
import numpy as np


def generate_anchors(image_size=[640, 640], ):
    """Generate anchor boxes based on image size."""
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
        for i, j in product(range(map_height), range(map_width)):
            for min_size in min_sizes[k]:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]

                dense_cx = [x * step / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * step / image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]

    output = torch.Tensor(anchors).view(-1, 4)
    return output


def nms(dets, threshold):
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


def decode(loc, priors, variances=[0.1, 0.2]):
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


def decode_landmarks(predictions, priors, variances=[0.1, 0.2]):
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


def draw_detections(original_image, detections, vis_threshold):
    """
    Draws bounding boxes and landmarks on the image based on multiple detections.

    Args:
        original_image (ndarray): The image on which to draw detections.
        detections (ndarray): Array of detected bounding boxes and landmarks.
        vis_threshold (float): The confidence threshold for displaying detections.
    """

    # Colors for visualization
    LANDMARK_COLORS = [
        (0, 0, 255),    # Right eye (Red)
        (0, 255, 255),  # Left eye (Yellow)
        (255, 0, 255),  # Nose (Magenta)
        (0, 255, 0),    # Right mouth (Green)
        (255, 0, 0)     # Left mouth (Blue)
    ]
    BOX_COLOR = (0, 0, 255)
    TEXT_COLOR = (255, 255, 255)

    # Filter by confidence
    detections = detections[detections[:, 4] >= vis_threshold]

    print(f"#faces: {len(detections)}")

    # Slice arrays efficiently
    boxes = detections[:, 0:4].astype(np.int32)
    scores = detections[:, 4]
    landmarks = detections[:, 5:15].reshape(-1, 5, 2).astype(np.int32)

    for box, score, landmark in zip(boxes, scores, landmarks):
        # Draw bounding box
        cv2.rectangle(original_image, (box[0], box[1]), (box[2], box[3]), BOX_COLOR, 2)

        # Draw confidence score
        text = f"{score:.2f}"
        cx, cy = box[0], box[1] + 12
        cv2.putText(original_image, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, TEXT_COLOR)

        # Draw landmarks
        for point, color in zip(landmark, LANDMARK_COLORS):
            cv2.circle(original_image, point, 1, color, 4)
