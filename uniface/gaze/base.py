# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from uniface.types import GazeResult

__all__ = ['BaseGazeEstimator', 'GazeResult']


class BaseGazeEstimator(ABC):
    """
    Abstract base class for all gaze estimation models.

    This class defines the common interface that all gaze estimators must implement,
    ensuring consistency across different gaze estimation methods. Gaze estimation
    predicts the direction a person is looking based on their face image.

    The gaze direction is represented as pitch and yaw angles in radians:
    - Pitch: Vertical angle (positive = looking up, negative = looking down)
    - Yaw: Horizontal angle (positive = looking right, negative = looking left)
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
    def postprocess(self, outputs: tuple[np.ndarray, np.ndarray]) -> GazeResult:
        """
        Postprocess raw model outputs into gaze angles.

        This method takes the raw output from the model's inference and
        converts it into pitch and yaw angles in radians.

        Args:
            outputs: Raw outputs from the model inference. The format depends
                     on the specific model architecture.

        Returns:
            GazeResult: Result containing pitch and yaw angles in radians.
        """
        raise NotImplementedError('Subclasses must implement the postprocess method.')

    @abstractmethod
    def estimate(self, face_image: np.ndarray) -> GazeResult:
        """
        Perform end-to-end gaze estimation on a face image.

        This method orchestrates the full pipeline: preprocessing the input,
        running inference, and postprocessing to return the gaze direction.

        Args:
            face_image (np.ndarray): A cropped face image in BGR format.
                                     The face should be roughly centered and
                                     well-framed within the image.

        Returns:
            GazeResult: Result containing pitch and yaw angles in radians:
                - pitch: Vertical gaze angle (positive = up, negative = down)
                - yaw: Horizontal gaze angle (positive = right, negative = left)

        Example:
            >>> estimator = create_gaze_estimator()
            >>> result = estimator.estimate(face_crop)
            >>> print(f'Looking: pitch={np.degrees(result.pitch):.1f}°, yaw={np.degrees(result.yaw):.1f}°')
        """
        raise NotImplementedError('Subclasses must implement the estimate method.')

    def __call__(self, face_image: np.ndarray) -> GazeResult:
        """
        Provides a convenient, callable shortcut for the `estimate` method.

        Args:
            face_image (np.ndarray): A cropped face image in BGR format.

        Returns:
            GazeResult: Result containing pitch and yaw angles in radians.
        """
        return self.estimate(face_image)
