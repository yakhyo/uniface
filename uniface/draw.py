# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import colorsys
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from uniface.types import Face

__all__ = [
    'FACE_PARSING_COLORS',
    'FACE_PARSING_LABELS',
    'calculate_optimal_line_thickness',
    'calculate_optimal_text_scale',
    'draw_corner_bbox',
    'draw_detections',
    'draw_gaze',
    'draw_text_label',
    'draw_tracks',
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

# Per-point colors for 5-point facial landmarks (BGR)
_LANDMARK_COLORS = (
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
)


def _get_color(idx: int) -> tuple[int, int, int]:
    """Get a visually distinct BGR color for a given index.

    Uses golden-ratio hue stepping in HSV space to maximize perceptual
    separation between consecutive indices. Works for any non-negative index.

    Args:
        idx: Non-negative integer index (e.g. track ID).

    Returns:
        BGR color tuple suitable for OpenCV drawing functions.
    """
    golden_ratio = 0.618033988749895
    hue = (idx * golden_ratio) % 1.0
    # HSV -> RGB with fixed saturation=0.85 and value=0.95 for vivid colors
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def calculate_optimal_line_thickness(resolution_wh: tuple[int, int]) -> int:
    """Calculate adaptive line thickness based on image resolution.

    Args:
        resolution_wh: Image resolution as ``(width, height)``.

    Returns:
        Recommended line thickness in pixels.

    Example:
        >>> calculate_optimal_line_thickness((1920, 1080))
        4
        >>> calculate_optimal_line_thickness((640, 480))
        2
    """
    return max(round(sum(resolution_wh) / 2 * 0.003), 2)


def calculate_optimal_text_scale(resolution_wh: tuple[int, int]) -> float:
    """Calculate adaptive font scale based on image resolution.

    Args:
        resolution_wh: Image resolution as ``(width, height)``.

    Returns:
        Recommended font scale factor.

    Example:
        >>> calculate_optimal_text_scale((1920, 1080))
        1.08
        >>> calculate_optimal_text_scale((640, 480))
        0.48
    """
    return min(resolution_wh) * 1e-3


def draw_corner_bbox(
    image: np.ndarray,
    bbox: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
    proportion: float = 0.2,
) -> None:
    """Draw a bounding box with corner brackets on an image.

    Draws a thin full rectangle with thick corner accents, commonly used in
    face-detection overlays for a clean look.

    Args:
        image: Input image to draw on (modified in-place).
        bbox: Bounding box in xyxy format ``[x1, y1, x2, y2]``.
        color: BGR color of the box. Defaults to green ``(0, 255, 0)``.
        thickness: Thickness of corner bracket lines. Defaults to 3.
        proportion: Corner length as a fraction of the shorter side.
            Defaults to 0.2.
    """
    x1, y1, x2, y2 = map(int, bbox)
    corner_length = int(proportion * min(x2 - x1, y2 - y1))

    # Thin full rectangle
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


def draw_text_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    bg_color: tuple[int, int, int],
    text_color: tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.5,
    font_thickness: int = 2,
    padding: int = 5,
) -> None:
    """Draw text with a filled background rectangle above a given position.

    The label is placed so that its bottom edge sits at *y*, making it
    suitable for positioning above a bounding box top-left corner.

    Args:
        image: Input image to draw on (modified in-place).
        text: The text string to render.
        x: Left x-coordinate for the label.
        y: Bottom y-coordinate for the label (e.g. ``bbox[1]``).
        bg_color: BGR background fill color.
        text_color: BGR text color. Defaults to white.
        font_scale: OpenCV font scale factor. Defaults to 0.5.
        font_thickness: OpenCV font thickness. Defaults to 2.
        padding: Pixel padding around the text. Defaults to 5.
    """
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    cv2.rectangle(
        image,
        (x, y - th - baseline - padding * 2),
        (x + tw + padding * 2, y),
        bg_color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + padding, y - padding),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        font_thickness,
    )


def draw_detections(
    *,
    image: np.ndarray,
    bboxes: list[np.ndarray] | list[list[float]],
    scores: np.ndarray | list[float],
    landmarks: list[np.ndarray] | list[list[list[float]]],
    vis_threshold: float = 0.6,
    draw_score: bool = False,
    corner_bbox: bool = True,
) -> None:
    """Draw bounding boxes, landmarks, and optional scores on an image.

    Modifies the image in-place.

    Args:
        image: Input image to draw on (modified in-place).
        bboxes: List of bounding boxes in xyxy format ``[x1, y1, x2, y2]``.
        scores: List of confidence scores.
        landmarks: List of landmark sets with shape ``(5, 2)``.
        vis_threshold: Confidence threshold for filtering. Defaults to 0.6.
        draw_score: Whether to draw confidence scores. Defaults to False.
        corner_bbox: Use corner-style bounding boxes. Defaults to True.
    """

    # Adaptive line thickness
    line_thickness = max(round(sum(image.shape[:2]) / 2 * 0.003), 2)

    for i, score in enumerate(scores):
        if score < vis_threshold:
            continue

        bbox = np.array(bboxes[i], dtype=np.int32)

        # Draw bounding box
        if corner_bbox:
            draw_corner_bbox(image, bbox, color=(0, 255, 0), thickness=line_thickness, proportion=0.2)
        else:
            cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), (0, 255, 0), line_thickness)

        # Draw confidence score label
        if draw_score:
            font_scale = max(0.4, min(0.7, (bbox[3] - bbox[1]) / 200))
            draw_text_label(
                image,
                f'{score:.2f}',
                bbox[0],
                bbox[1],
                bg_color=(0, 255, 0),
                text_color=(0, 0, 0),
                font_scale=font_scale,
            )

        # Draw landmarks
        landmark_set = np.array(landmarks[i], dtype=np.int32)
        for j, point in enumerate(landmark_set):
            cv2.circle(image, tuple(point), line_thickness + 1, _LANDMARK_COLORS[j % len(_LANDMARK_COLORS)], -1)


def draw_gaze(
    image: np.ndarray,
    bbox: np.ndarray,
    pitch: np.ndarray | float,
    yaw: np.ndarray | float,
    *,
    draw_bbox: bool = True,
    corner_bbox: bool = True,
    draw_angles: bool = True,
) -> None:
    """Draw gaze direction with optional bounding box on an image.

    Modifies the image in-place.

    Args:
        image: Input image to draw on (modified in-place).
        bbox: Face bounding box in xyxy format ``[x1, y1, x2, y2]``.
        pitch: Vertical gaze angle in radians.
        yaw: Horizontal gaze angle in radians.
        draw_bbox: Whether to draw the bounding box. Defaults to True.
        corner_bbox: Use corner-style bounding box. Defaults to True.
        draw_angles: Whether to display pitch/yaw values as text. Defaults to True.
    """

    x_min, y_min, x_max, y_max = map(int, bbox[:4])

    # Adaptive line thickness
    line_thickness = max(round(sum(image.shape[:2]) / 2 * 0.003), 2)

    # Draw bounding box if requested
    if draw_bbox:
        if corner_bbox:
            draw_corner_bbox(image, bbox, color=(0, 255, 0), thickness=line_thickness)
        else:
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), line_thickness)

    # Calculate center of the bounding box
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2

    # Calculate the direction of the gaze
    length = x_max - x_min
    dx = int(-length * np.sin(pitch) * np.cos(yaw))
    dy = int(-length * np.sin(yaw))

    # Draw gaze arrow
    center_radius = max(line_thickness + 1, 4)
    cv2.circle(image, (x_center, y_center), radius=center_radius, color=(0, 0, 255), thickness=-1)
    cv2.arrowedLine(
        image,
        (x_center, y_center),
        (x_center + dx, y_center + dy),
        color=(0, 0, 255),
        thickness=line_thickness,
        line_type=cv2.LINE_AA,
        tipLength=0.25,
    )

    # Draw angle values
    if draw_angles:
        font_scale = max(0.4, min(0.7, (y_max - y_min) / 200))
        draw_text_label(
            image,
            f'P:{np.degrees(pitch):.0f}deg Y:{np.degrees(yaw):.0f}deg',
            x_min,
            y_min,
            bg_color=(0, 0, 255),
            text_color=(255, 255, 255),
            font_scale=font_scale,
        )


def draw_tracks(
    *,
    image: np.ndarray,
    faces: list[Face],
    draw_landmarks: bool = True,
    draw_id: bool = True,
    corner_bbox: bool = True,
) -> None:
    """Draw tracked faces with color-coded track IDs on an image.

    Each track ID is assigned a deterministic color for consistent visualization
    across frames. Faces without a ``track_id`` are drawn in gray.

    Modifies the image in-place.

    Args:
        image: Input image to draw on (modified in-place).
        faces: List of Face objects (with ``track_id`` assigned by FaceTracker).
        draw_landmarks: Whether to draw facial landmarks. Defaults to True.
        draw_id: Whether to draw track ID labels. Defaults to True.
        corner_bbox: Use corner-style bounding boxes. Defaults to True.

    Example:
        >>> from uniface import FaceTracker, RetinaFace
        >>> from uniface.draw import draw_tracks
        >>> tracker = FaceTracker(RetinaFace())
        >>> faces = tracker.update(frame)
        >>> draw_tracks(image=frame, faces=faces)
    """
    untracked_color = (128, 128, 128)

    # Adaptive line thickness
    line_thickness = max(round(sum(image.shape[:2]) / 2 * 0.003), 2)

    for face in faces:
        bbox = np.array(face.bbox, dtype=np.int32)
        track_id = face.track_id

        # Pick color based on track ID
        color = _get_color(track_id) if track_id is not None else untracked_color

        # Draw bounding box
        if corner_bbox:
            draw_corner_bbox(image, bbox, color=color, thickness=line_thickness, proportion=0.2)
        else:
            cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), color, line_thickness)

        # Draw track ID label
        if draw_id and track_id is not None:
            font_scale = max(0.4, min(0.7, (bbox[3] - bbox[1]) / 200))
            draw_text_label(
                image,
                f'ID:{track_id}',
                bbox[0],
                bbox[1],
                bg_color=color,
                font_scale=font_scale,
            )

        # Draw landmarks
        if draw_landmarks and face.landmarks is not None:
            landmark_set = np.array(face.landmarks, dtype=np.int32)
            for j, point in enumerate(landmark_set):
                cv2.circle(image, tuple(point), line_thickness + 1, _LANDMARK_COLORS[j % len(_LANDMARK_COLORS)], -1)


def vis_parsing_maps(
    image: np.ndarray,
    segmentation_mask: np.ndarray,
    *,
    save_image: bool = False,
    save_path: str = 'result.png',
) -> np.ndarray:
    """Visualize face parsing segmentation mask by overlaying colored regions.

    Args:
        image: Input face image in RGB format with shape ``(H, W, 3)``.
        segmentation_mask: Segmentation mask with shape ``(H, W)`` where each
            pixel value represents a facial component class (0-18).
        save_image: Whether to save the visualization to disk. Defaults to False.
        save_path: Path to save the visualization if *save_image* is True.

    Returns:
        Blended image with segmentation overlay in BGR format.

    Example:
        >>> import cv2
        >>> from uniface.parsing import BiSeNet
        >>> from uniface.draw import vis_parsing_maps
        >>> parser = BiSeNet()
        >>> face_image = cv2.imread('face.jpg')
        >>> mask = parser.parse(face_image)
        >>> face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        >>> result = vis_parsing_maps(face_rgb, mask)
        >>> cv2.imwrite('parsed_face.jpg', result)
    """
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
    blended_image = cv2.addWeighted(bgr_image, 0.6, segmentation_mask_color, 0.4, 0)

    if save_image:
        cv2.imwrite(save_path, blended_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return blended_image
