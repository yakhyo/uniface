# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from typing import List, Union

import cv2
import numpy as np


def draw_detections(
    image: np.ndarray,
    bboxes: Union[List[np.ndarray], List[List[float]]],
    scores: Union[np.ndarray, List[float]],
    landmarks: Union[List[np.ndarray], List[List[List[float]]]],
    vis_threshold: float = 0.6,
):
    """
    Draws bounding boxes, scores, and landmarks from separate lists onto an image.

    Args:
        image (np.ndarray): The image to draw on.
        bboxes (List[np.ndarray] or List[List[float]]): List of bounding boxes. Each bbox can be
            np.ndarray with shape (4,) or list [x1, y1, x2, y2].
        scores (List[float] or np.ndarray): List or array of confidence scores.
        landmarks (List[np.ndarray] or List[List[List[float]]]): List of landmark sets. Each landmark
            set can be np.ndarray with shape (5, 2) or nested list [[[x,y],...],...].
        vis_threshold (float): Confidence threshold for filtering which detections to draw.
    """
    _colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 0, 0)]

    # Filter detections by score
    keep_indices = [i for i, score in enumerate(scores) if score >= vis_threshold]

    # Draw the filtered detections
    for i in keep_indices:
        bbox = np.array(bboxes[i], dtype=np.int32)
        score = scores[i]
        landmark_set = np.array(landmarks[i], dtype=np.int32)

        # Calculate adaptive thickness
        thickness = max(1, int(min(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 100))

        # Draw bounding box
        cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), (0, 0, 255), thickness)

        # Draw score
        cv2.putText(
            image,
            f'{score:.2f}',
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            thickness,
        )

        # Draw landmarks
        for j, point in enumerate(landmark_set):
            cv2.circle(image, tuple(point), thickness + 1, _colors[j], -1)
