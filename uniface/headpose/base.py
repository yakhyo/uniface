# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from uniface.types import HeadPoseResult

__all__ = ['BaseHeadPoseEstimator', 'HeadPoseResult']


class BaseHeadPoseEstimator(ABC):
    """
    Abstract base class for all head pose estimation models.

    This class defines the common interface that all head pose estimators must implement,
    ensuring consistency across different head pose estimation methods. Head pose estimation
    predicts the orientation of a person's head based on their face image.

    The head orientation is represented as Euler angles in degrees:
    - Pitch: Rotation around X-axis (positive = looking down, negative = looking up)
    - Yaw: Rotation around Y-axis (positive = looking right, negative = looking left)
    - Roll: Rotation around Z-axis (positive = tilting clockwise, negative = tilting counter-clockwise)
    """

    @abstractmethod
    def _initialize_model(self) -> None:
        """
        Initialize the underlying model for inference.

        This method should handle loading model weights, creating the
        inference session (e.g., ONNX Runtime), and any necessary
        setup procedures to prepare the model for prediction.

        Raises:
            RuntimeError: If the model fails to load or initialize.
        """
        raise NotImplementedError('Subclasses must implement the _initialize_model method.')

    @abstractmethod
    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input face image for model inference.

        This method should take a raw face crop and convert it into the format
        expected by the model's inference engine (e.g., normalized tensor).

        Args:
            face_image (np.ndarray): A cropped face image in BGR format with
                                     shape (H, W, C).

        Returns:
            np.ndarray: The preprocessed image tensor ready for inference,
                        typically with shape (1, C, H, W).
        """
        raise NotImplementedError('Subclasses must implement the preprocess method.')

    @abstractmethod
    def postprocess(self, rotation_matrix: np.ndarray) -> HeadPoseResult:
        """
        Postprocess a rotation matrix into Euler angles.

        This method takes the raw rotation matrix output from the model's
        inference and converts it into pitch, yaw, and roll angles in degrees.

        Args:
            rotation_matrix: Rotation matrix with shape (B, 3, 3) from the
                             model inference.

        Returns:
            HeadPoseResult: Result containing pitch, yaw, and roll in degrees.
        """
        raise NotImplementedError('Subclasses must implement the postprocess method.')

    @abstractmethod
    def estimate(self, face_image: np.ndarray) -> HeadPoseResult:
        """
        Perform end-to-end head pose estimation on a face image.

        This method orchestrates the full pipeline: preprocessing the input,
        running inference, and postprocessing to return the head orientation.

        Args:
            face_image (np.ndarray): A cropped face image in BGR format.
                                     The face should be roughly centered and
                                     well-framed within the image.

        Returns:
            HeadPoseResult: Result containing Euler angles in degrees:
                - pitch: Rotation around X-axis (positive = down)
                - yaw: Rotation around Y-axis (positive = right)
                - roll: Rotation around Z-axis (positive = clockwise)

        Example:
            >>> estimator = create_head_pose_estimator()
            >>> result = estimator.estimate(face_crop)
            >>> print(f'Pose: pitch={result.pitch:.1f}°, yaw={result.yaw:.1f}°, roll={result.roll:.1f}°')
        """
        raise NotImplementedError('Subclasses must implement the estimate method.')

    def __call__(self, face_image: np.ndarray) -> HeadPoseResult:
        """
        Provides a convenient, callable shortcut for the `estimate` method.

        Args:
            face_image (np.ndarray): A cropped face image in BGR format.

        Returns:
            HeadPoseResult: Result containing pitch, yaw, and roll in degrees.
        """
        return self.estimate(face_image)
