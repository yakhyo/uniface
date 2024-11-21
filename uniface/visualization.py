# Copyright 2024 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import cv2
import numpy as np


def draw_detections(image, detections, vis_threshold=0.6):
    """
    Draw bounding boxes and landmarks on the image.

    Args:
        image (ndarray): Image to draw detections on.
        detections (tuple): (bounding boxes, landmarks) as NumPy arrays.
        vis_threshold (float): Confidence threshold for filtering detections.
    """

    _colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 0, 0)]

    # Unpack detections
    boxes, landmarks = detections
    scores = boxes[:, 4]

    # Filter detections by confidence threshold
    filtered = scores >= vis_threshold
    boxes = boxes[filtered, :4].astype(np.int32)
    landmarks = landmarks[filtered]
    scores = scores[filtered]

    # Draw bounding boxes, scores, and landmarks
    for box, score, landmark in zip(boxes, scores, landmarks):
        cv2.rectangle(image, box[:2], box[2:], (0, 0, 255), 2)
        cv2.putText(image, f"{score:.2f}", (box[0], box[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for point, color in zip(landmark, _colors):
            cv2.circle(image, tuple(point), 2, color, -1)
