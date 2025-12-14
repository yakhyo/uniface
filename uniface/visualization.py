# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from typing import List, Tuple, Union

import cv2
import numpy as np


def draw_detections(
    *,
    image: np.ndarray,
    bboxes: Union[List[np.ndarray], List[List[float]]],
    scores: Union[np.ndarray, List[float]],
    landmarks: Union[List[np.ndarray], List[List[List[float]]]],
    vis_threshold: float = 0.6,
    draw_score: bool = False,
    fancy_bbox: bool = True,
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
        if fancy_bbox:
            draw_fancy_bbox(image, bbox, color=(0, 255, 0), thickness=line_thickness, proportion=0.2)
        else:
            cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), (0, 255, 0), line_thickness)

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


def draw_fancy_bbox(
    image: np.ndarray,
    bbox: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
    proportion: float = 0.2,
):
    """
    Draws a bounding box with fancy corners on an image.

    Args:
        image: Input image to draw on.
        bbox: Bounding box coordinates [x1, y1, x2, y2].
        color: Color of the bounding box. Defaults to green.
        thickness: Thickness of the bounding box lines. Defaults to 3.
        proportion: Proportion of the corner length to the width/height of the bounding box. Defaults to 0.2.
    """
    x1, y1, x2, y2 = map(int, bbox)
    width = x2 - x1
    height = y2 - y1

    corner_length = int(proportion * min(width, height))

    # Draw the rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

    # Top-left corner
    cv2.line(image, (x1, y1), (x1 + corner_length, y1), color, thickness)
    cv2.line(image, (x1, y1), (x1, y1 + corner_length), color, thickness)

    # Top-right corner
    cv2.line(image, (x2, y1), (x2 - corner_length, y1), color, thickness)
    cv2.line(image, (x2, y1), (x2, y1 + corner_length), color, thickness)

    # Bottom-left corner
    cv2.line(image, (x1, y2), (x1, y2 - corner_length), color, thickness)
    cv2.line(image, (x1, y2), (x1 + corner_length, y2), color, thickness)

    # Bottom-right corner
    cv2.line(image, (x2, y2), (x2, y2 - corner_length), color, thickness)
    cv2.line(image, (x2, y2), (x2 - corner_length, y2), color, thickness)


def draw_gaze(
    image: np.ndarray,
    bbox: np.ndarray,
    pitch: np.ndarray,
    yaw: np.ndarray,
    *,
    draw_bbox: bool = True,
    fancy_bbox: bool = True,
    draw_angles: bool = True,
):
    """
    Draws gaze direction with optional bounding box on an image.

    Args:
        image: Input image to draw on (modified in-place).
        bbox: Face bounding box [x1, y1, x2, y2].
        pitch: Vertical gaze angle in radians.
        yaw: Horizontal gaze angle in radians.
        draw_bbox: Whether to draw the bounding box. Defaults to True.
        fancy_bbox: Use fancy corner-style bbox. Defaults to True.
        draw_angles: Whether to display pitch/yaw values as text. Defaults to False.
    """
    x_min, y_min, x_max, y_max = map(int, bbox[:4])

    # Calculate dynamic line thickness based on image size (same as draw_detections)
    line_thickness = max(round(sum(image.shape[:2]) / 2 * 0.003), 2)

    # Calculate dynamic font scale based on bbox height (same as draw_detections)
    bbox_h = y_max - y_min
    font_scale = max(0.4, min(0.7, bbox_h / 200))
    font_thickness = 2

    # Draw bounding box if requested
    if draw_bbox:
        if fancy_bbox:
            draw_fancy_bbox(image, bbox, color=(0, 255, 0), thickness=line_thickness)
        else:
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), line_thickness)

    # Calculate center of the bounding box
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2

    # Calculate the direction of the gaze
    length = x_max - x_min
    dx = int(-length * np.sin(pitch) * np.cos(yaw))
    dy = int(-length * np.sin(yaw))

    point1 = (x_center, y_center)
    point2 = (x_center + dx, y_center + dy)

    # Calculate dynamic center point radius based on line thickness
    center_radius = max(line_thickness + 1, 4)

    # Draw gaze direction
    cv2.circle(image, (x_center, y_center), radius=center_radius, color=(0, 0, 255), thickness=-1)
    cv2.arrowedLine(
        image,
        point1,
        point2,
        color=(0, 0, 255),
        thickness=line_thickness,
        line_type=cv2.LINE_AA,
        tipLength=0.25,
    )

    # Draw angle values
    if draw_angles:
        text = f'P:{np.degrees(pitch):.0f}deg Y:{np.degrees(yaw):.0f}deg'
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        # Draw background rectangle for text
        cv2.rectangle(
            image,
            (x_min, y_min - text_height - baseline - 10),
            (x_min + text_width + 10, y_min),
            (0, 0, 255),
            -1,
        )

        # Draw text
        cv2.putText(
            image,
            text,
            (x_min + 5, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
        )
