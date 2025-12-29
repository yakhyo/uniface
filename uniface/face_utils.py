# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import cv2
import numpy as np
from skimage.transform import SimilarityTransform

__all__ = [
    'bbox_center_alignment',
    'compute_similarity',
    'face_alignment',
    'transform_points_2d',
]


# Standard 5-point facial landmark reference for ArcFace alignment (112x112)
reference_alignment: np.ndarray = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def estimate_norm(
    landmark: np.ndarray,
    image_size: int | tuple[int, int] = 112,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the normalization transformation matrix for facial landmarks.

    Args:
        landmark: Array of shape (5, 2) representing the coordinates of the facial landmarks.
        image_size: The size of the output image. Can be an integer (for square images)
            or a tuple (width, height). Default is 112.

    Returns:
        A tuple containing:
            - The 2x3 transformation matrix for aligning the landmarks.
            - The 2x3 inverse transformation matrix.

    Raises:
        AssertionError: If the input landmark array does not have the shape (5, 2)
            or if image_size is not a multiple of 112 or 128.
    """
    assert landmark.shape == (5, 2), 'Landmark array must have shape (5, 2).'

    # Handle both int and tuple inputs
    if isinstance(image_size, tuple):
        size = image_size[0]  # Use width for ratio calculation
    else:
        size = image_size

    assert size % 112 == 0 or size % 128 == 0, 'Image size must be a multiple of 112 or 128.'

    if size % 112 == 0:
        ratio = float(size) / 112.0
        diff_x = 0.0
    else:
        ratio = float(size) / 128.0
        diff_x = 8.0 * ratio

    # Adjust reference alignment based on ratio and diff_x
    alignment = reference_alignment * ratio
    alignment[:, 0] += diff_x

    # Compute the transformation matrix
    transform = SimilarityTransform()
    transform.estimate(landmark, alignment)

    matrix = transform.params[0:2, :]
    inverse_matrix = np.linalg.inv(transform.params)[0:2, :]

    return matrix, inverse_matrix


def face_alignment(
    image: np.ndarray,
    landmark: np.ndarray,
    image_size: int | tuple[int, int] = 112,
) -> tuple[np.ndarray, np.ndarray]:
    """Align the face in the input image based on the given facial landmarks.

    Args:
        image: Input image as a NumPy array with shape (H, W, C).
        landmark: Array of shape (5, 2) representing the facial landmark coordinates.
        image_size: The size of the aligned output image. Can be an integer
            (for square images) or a tuple (width, height). Default is 112.

    Returns:
        A tuple containing:
            - The aligned face as a NumPy array.
            - The 2x3 inverse transformation matrix used for alignment.
    """
    # Get the transformation matrix
    transform_matrix, inverse_transform = estimate_norm(landmark, image_size)

    # Handle both int and tuple for warpAffine output size
    if isinstance(image_size, int):
        output_size = (image_size, image_size)
    else:
        output_size = image_size

    # Warp the input image to align the face
    warped = cv2.warpAffine(image, transform_matrix, output_size, borderValue=0.0)

    return warped, inverse_transform


def compute_similarity(feat1: np.ndarray, feat2: np.ndarray, normalized: bool = False) -> np.float32:
    """Compute cosine similarity between two face embeddings.

    Args:
        feat1: First embedding vector.
        feat2: Second embedding vector.
        normalized: Set True if the embeddings are already L2 normalized.

    Returns:
        Cosine similarity score in range [-1, 1].
    """
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    if normalized:
        return np.dot(feat1, feat2)
    # Add small epsilon to prevent division by zero
    return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-5)


def bbox_center_alignment(
    image: np.ndarray,
    center: tuple[float, float],
    output_size: int,
    scale: float,
    rotation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply center-based alignment, scaling, and rotation to an image.

    Args:
        image: Input image with shape (H, W, C).
        center: Center point (x, y), e.g., face center from bbox.
        output_size: Desired output image size (square).
        scale: Scaling factor to zoom in/out.
        rotation: Rotation angle in degrees (clockwise).

    Returns:
        A tuple containing:
            - Aligned and cropped image with shape (output_size, output_size, C).
            - 2x3 affine transform matrix used.
    """

    # Convert rotation from degrees to radians
    rot = float(rotation) * np.pi / 180.0

    # Scale the image
    t1 = SimilarityTransform(scale=scale)

    # Translate the center point to the origin (after scaling)
    cx = center[0] * scale
    cy = center[1] * scale
    t2 = SimilarityTransform(translation=(-1 * cx, -1 * cy))

    # Apply rotation around origin (center of face)
    t3 = SimilarityTransform(rotation=rot)

    # Translate origin to center of output image
    t4 = SimilarityTransform(translation=(output_size / 2, output_size / 2))

    # Combine all transformations in order: scale → center shift → rotate → recentralize
    t = t1 + t2 + t3 + t4

    # Extract 2x3 affine matrix
    M = t.params[0:2]

    # Warp the image using OpenCV
    cropped = cv2.warpAffine(image, M, (output_size, output_size), borderValue=0.0)

    return cropped, M


def transform_points_2d(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 2D affine transformation to an array of 2D points.

    Args:
        points: An (N, 2) array of 2D points.
        transform: A (2, 3) affine transformation matrix.

    Returns:
        Transformed (N, 2) array of points.
    """
    transformed = np.zeros_like(points, dtype=np.float32)
    for i in range(points.shape[0]):
        point = np.array([points[i, 0], points[i, 1], 1.0], dtype=np.float32)
        result = np.dot(transform, point)
        transformed[i] = result[:2]

    return transformed
