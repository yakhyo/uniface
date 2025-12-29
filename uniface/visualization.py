# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Visualization utilities for UniFace.

This module provides functions for drawing detection results, gaze directions,
and face parsing segmentation maps on images.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    'FACE_PARSING_COLORS',
    'FACE_PARSING_LABELS',
    'draw_detections',
    'draw_fancy_bbox',
    'draw_gaze',
    'vis_parsing_maps',
]

# Face parsing component names (19 classes)
FACE_PARSING_LABELS = [
    'background',
    'skin',
    'l_brow',
    'r_brow',
    'l_eye',
    'r_eye',
    'eye_g',
    'l_ear',
    'r_ear',
    'ear_r',
    'nose',
    'mouth',
    'u_lip',
    'l_lip',
    'neck',
    'neck_l',
    'cloth',
    'hair',
    'hat',
]

# Color palette for face parsing visualization
FACE_PARSING_COLORS = [
    [0, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 0, 85],
    [255, 0, 170],
    [0, 255, 0],
    [85, 255, 0],
    [170, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [0, 85, 255],
    [0, 170, 255],
    [255, 255, 0],
    [255, 255, 85],
    [255, 255, 170],
    [255, 0, 255],
]


def draw_detections(
    *,
    image: np.ndarray,
    bboxes: list[np.ndarray] | list[list[float]],
    scores: np.ndarray | list[float],
    landmarks: list[np.ndarray] | list[list[list[float]]],
    vis_threshold: float = 0.6,
    draw_score: bool = False,
    fancy_bbox: bool = True,
) -> None:
    """Draw bounding boxes, landmarks, and optional scores on an image.

    Modifies the image in-place.

    Args:
        image: Input image to draw on (modified in-place).
        bboxes: List of bounding boxes as [x1, y1, x2, y2].
        scores: List of confidence scores.
        landmarks: List of landmark sets with shape (5, 2).
        vis_threshold: Confidence threshold for filtering. Defaults to 0.6.
        draw_score: Whether to draw confidence scores. Defaults to False.
        fancy_bbox: Use corner-style bounding boxes. Defaults to True.
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
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
    proportion: float = 0.2,
) -> None:
    """Draw a bounding box with fancy corners on an image.

    Args:
        image: Input image to draw on (modified in-place).
        bbox: Bounding box coordinates [x1, y1, x2, y2].
        color: Color of the bounding box in BGR. Defaults to green.
        thickness: Thickness of the corner lines. Defaults to 3.
        proportion: Proportion of corner length to box dimensions. Defaults to 0.2.
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
    pitch: np.ndarray | float,
    yaw: np.ndarray | float,
    *,
    draw_bbox: bool = True,
    fancy_bbox: bool = True,
    draw_angles: bool = True,
) -> None:
    """Draw gaze direction with optional bounding box on an image.

    Args:
        image: Input image to draw on (modified in-place).
        bbox: Face bounding box [x1, y1, x2, y2].
        pitch: Vertical gaze angle in radians.
        yaw: Horizontal gaze angle in radians.
        draw_bbox: Whether to draw the bounding box. Defaults to True.
        fancy_bbox: Use fancy corner-style bbox. Defaults to True.
        draw_angles: Whether to display pitch/yaw values as text. Defaults to True.
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


def vis_parsing_maps(
    image: np.ndarray,
    segmentation_mask: np.ndarray,
    *,
    save_image: bool = False,
    save_path: str = 'result.png',
) -> np.ndarray:
    """Visualize face parsing segmentation mask by overlaying colored regions.

    Args:
        image: Input face image in RGB format with shape (H, W, 3).
        segmentation_mask: Segmentation mask with shape (H, W) where each pixel
            value represents a facial component class (0-18).
        save_image: Whether to save the visualization to disk. Defaults to False.
        save_path: Path to save the visualization if save_image is True.

    Returns:
        Blended image with segmentation overlay in BGR format.

    Example:
        >>> import cv2
        >>> from uniface.parsing import BiSeNet
        >>> from uniface.visualization import vis_parsing_maps
        >>> parser = BiSeNet()
        >>> face_image = cv2.imread('face.jpg')
        >>> mask = parser.parse(face_image)
        >>> face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        >>> result = vis_parsing_maps(face_rgb, mask)
        >>> cv2.imwrite('parsed_face.jpg', result)
    """
    # Create numpy arrays for image and segmentation mask
    image = np.array(image).copy().astype(np.uint8)
    segmentation_mask = segmentation_mask.copy().astype(np.uint8)

    # Create a color mask
    segmentation_mask_color = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1], 3))

    num_classes = np.max(segmentation_mask)

    for class_index in range(1, num_classes + 1):
        class_pixels = np.where(segmentation_mask == class_index)
        segmentation_mask_color[class_pixels[0], class_pixels[1], :] = FACE_PARSING_COLORS[class_index]

    segmentation_mask_color = segmentation_mask_color.astype(np.uint8)

    # Convert image to BGR format for blending
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Blend the image with the segmentation mask
    blended_image = cv2.addWeighted(bgr_image, 0.6, segmentation_mask_color, 0.4, 0)

    # Save the result if required
    if save_image:
        cv2.imwrite(save_path, blended_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return blended_image
