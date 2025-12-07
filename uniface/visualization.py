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
    draw_score: bool = False,
):
    """
    Draws bounding boxes, landmarks, and optional scores on an image.

    Args:
        image: Input image to draw on.
        bboxes: List of bounding boxes [x1, y1, x2, y2].
        scores: List of confidence scores.
        landmarks: List of landmark sets with shape (5, 2).
        vis_threshold: Confidence threshold for filtering. Defaults to 0.6.
        draw_score: Whether to draw confidence scores. Defaults to False.
    """
    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 0, 0)]

    # Calculate line thickness based on image size
    line_thickness = max(round(sum(image.shape[:2]) / 2 * 0.003), 2)

    # Filter detections by confidence threshold
    keep_indices = [i for i, score in enumerate(scores) if score >= vis_threshold]

    for i in keep_indices:
        bbox = np.array(bboxes[i], dtype=np.int32)
        score = scores[i]
        landmark_set = np.array(landmarks[i], dtype=np.int32)

        # Calculate dynamic font scale based on bbox height
        bbox_h = bbox[3] - bbox[1]
        font_scale = max(0.4, min(0.7, bbox_h / 200))
        font_thickness = 2

        # Draw bounding box
        cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), (0, 0, 255), line_thickness)

        # Draw confidence score with background
        if draw_score:
            text = f'{score:.2f}'
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            # Draw background rectangle
            cv2.rectangle(
                image,
                (bbox[0], bbox[1] - text_height - baseline - 10),
                (bbox[0] + text_width + 10, bbox[1]),
                (0, 255, 0),
                -1,
            )

            # Draw text
            cv2.putText(
                image,
                text,
                (bbox[0] + 5, bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                font_thickness,
            )

        # Draw landmarks
        for j, point in enumerate(landmark_set):
            cv2.circle(image, tuple(point), line_thickness + 1, colors[j], -1)
