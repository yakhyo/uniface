# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from uniface.types import SpoofingResult

__all__ = ['BaseSpoofer', 'SpoofingResult']


class BaseSpoofer(ABC):
    """
    Abstract base class for all face anti-spoofing models.

    This class defines the common interface that all anti-spoofing models must implement,
    ensuring consistency across different spoofing detection methods. Anti-spoofing models
    detect whether a face is real (live person) or fake (photo, video, mask, etc.).
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
    def preprocess(self, image: np.ndarray, bbox: list | np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for model inference.

        This method should crop the face region using the bounding box,
        resize it to the model's expected input size, and normalize
        the pixel values as required by the model.

        Args:
            image (np.ndarray): Input image in BGR format with shape (H, W, C).
            bbox (Union[List, np.ndarray]): Face bounding box in [x1, y1, x2, y2] format.

        Returns:
            np.ndarray: The preprocessed image tensor ready for inference,
                        typically with shape (1, C, H, W).
        """
        raise NotImplementedError('Subclasses must implement the preprocess method.')

    @abstractmethod
    def postprocess(self, outputs: np.ndarray) -> SpoofingResult:
        """
        Postprocess raw model outputs into prediction result.

        This method takes the raw output from the model's inference and
        converts it into a SpoofingResult.

        Args:
            outputs (np.ndarray): Raw outputs from the model inference (logits).

        Returns:
            SpoofingResult: Result containing is_real flag and confidence score.
        """
        raise NotImplementedError('Subclasses must implement the postprocess method.')

    @abstractmethod
    def predict(self, image: np.ndarray, bbox: list | np.ndarray) -> SpoofingResult:
        """
        Perform end-to-end anti-spoofing prediction on a face.

        This method orchestrates the full pipeline: preprocessing the input,
        running inference, and postprocessing to return the prediction.

        Args:
            image (np.ndarray): Input image in BGR format containing the face.
            bbox (Union[List, np.ndarray]): Face bounding box in [x1, y1, x2, y2] format.
                This is typically obtained from a face detector.

        Returns:
            SpoofingResult: Result containing is_real flag and confidence score.

        Example:
            >>> spoofer = MiniFASNet()
            >>> detector = RetinaFace()
            >>> faces = detector.detect(image)
            >>> for face in faces:
            ...     result = spoofer.predict(image, face.bbox)
            ...     label = 'Real' if result.is_real else 'Fake'
            ...     print(f'{label}: {result.confidence:.2%}')
        """
        raise NotImplementedError('Subclasses must implement the predict method.')

    def __call__(self, image: np.ndarray, bbox: list | np.ndarray) -> SpoofingResult:
        """
        Provides a convenient, callable shortcut for the `predict` method.

        Args:
            image (np.ndarray): Input image in BGR format.
            bbox (Union[List, np.ndarray]): Face bounding box in [x1, y1, x2, y2] format.

        Returns:
            SpoofingResult: Result containing is_real flag and confidence score.
        """
        return self.predict(image, bbox)
