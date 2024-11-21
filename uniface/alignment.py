# Copyright 2024 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import cv2
import numpy as np
from skimage.transform import SimilarityTransform
from typing import Tuple

# Reference alignment for facial landmarks (ArcFace)
reference_alignment: np.ndarray = np.array(
    [[
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ]],
    dtype=np.float32
)


def estimate_norm(landmark: np.ndarray, image_size: int = 112) -> Tuple[np.ndarray, int]:
    """
    Estimate the normalization transformation matrix for facial landmarks.

    Args:
        landmark (np.ndarray): Array of shape (5, 2) representing the coordinates of the facial landmarks.
        image_size (int, optional): The size of the output image. Default is 112.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing:
            - min_matrix (np.ndarray): The 2x3 transformation matrix for aligning the landmarks.
            - min_index (int): The index of the reference alignment that resulted in the minimum error.

    Raises:
        AssertionError: If the input landmark array does not have the shape (5, 2).
    """
    assert landmark.shape == (5, 2), "Landmark array must have shape (5, 2)."
    min_matrix: np.ndarray = np.empty((2, 3))
    min_index: int = -1
    min_error: float = float('inf')

    # Prepare landmarks for transformation
    landmark_transform = np.insert(landmark, 2, values=np.ones(5), axis=1)
    transform = SimilarityTransform()

    # Adjust alignment based on image size
    if image_size == 112:
        alignment = reference_alignment
    else:
        alignment = (image_size / 112) * reference_alignment

    # Iterate through reference alignments
    for idx in np.arange(alignment.shape[0]):
        transform.estimate(landmark, alignment[idx])
        matrix = transform.params[0:2, :]
        results = np.dot(matrix, landmark_transform.T).T
        error = np.sum(np.sqrt(np.sum((results - alignment[idx]) ** 2, axis=1)))
        if error < min_error:
            min_error = error
            min_matrix = matrix
            min_index = idx

    return min_matrix, min_index


def face_alignment(image: np.ndarray, landmark: np.ndarray, image_size: int = 112) -> np.ndarray:
    """
    Align the face in the input image based on the given facial landmarks.

    Args:
        image (np.ndarray): Input image as a NumPy array.
        landmark (np.ndarray): Array of shape (5, 2) representing the coordinates of the facial landmarks.
        image_size (int, optional): The size of the aligned output image. Default is 112.

    Returns:
        np.ndarray: The aligned face as a NumPy array.
    """
    # Get the transformation matrix and pose index
    M, pose_index = estimate_norm(landmark, image_size)
    # Warp the input image to align the face
    warped = cv2.warpAffine(image, M, (image_size, image_size), borderValue=0.0)
    return warped
