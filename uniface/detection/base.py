# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from uniface.types import Face

__all__ = ['BaseDetector']


class BaseDetector(ABC):
    """Abstract base class for all face detectors.

    This class defines the interface that all face detectors must implement,
    ensuring consistency across different detection methods.

    Attributes:
        config: Dictionary containing detector configuration parameters.
        _supports_landmarks: Flag indicating if detector supports landmark detection.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the detector with configuration parameters.

        Args:
            **kwargs: Detector-specific configuration parameters.
        """
        self.config: dict[str, Any] = kwargs
        self._supports_landmarks: bool = False

    @abstractmethod
    def detect(self, image: np.ndarray, **kwargs: Any) -> list[Face]:
        """Detect faces in an image.

        Args:
            image: Input image as numpy array with shape (H, W, C) in BGR format.
            **kwargs: Additional detection parameters.

        Returns:
            List of detected Face objects, each containing:
                - bbox: Bounding box coordinates with shape (4,) as [x1, y1, x2, y2].
                - confidence: Detection confidence score (0.0 to 1.0).
                - landmarks: Facial landmarks with shape (5, 2) for 5-point landmarks.

        Example:
            >>> faces = detector.detect(image)
            >>> for face in faces:
            ...     bbox = face.bbox  # np.ndarray with shape (4,)
            ...     confidence = face.confidence  # float
            ...     landmarks = face.landmarks  # np.ndarray with shape (5, 2)
        """

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess input image for detection.

        Args:
            image: Input image with shape (H, W, C).

        Returns:
            Preprocessed image tensor ready for inference.
        """

    @abstractmethod
    def postprocess(self, outputs: Any, **kwargs: Any) -> Any:
        """Postprocess model outputs to get final detections.

        Args:
            outputs: Raw model outputs.
            **kwargs: Additional postprocessing parameters.

        Returns:
            Processed outputs (implementation-specific format, typically tuple of arrays).
        """

    def __str__(self) -> str:
        """String representation of the detector."""
        return f'{self.__class__.__name__}({self.config})'

    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()

    @property
    def supports_landmarks(self) -> bool:
        """Whether this detector supports landmark detection.

        Returns:
            True if landmarks are supported, False otherwise.
        """
        return hasattr(self, '_supports_landmarks') and self._supports_landmarks

    def get_info(self) -> dict[str, Any]:
        """Get detector information and configuration.

        Returns:
            Dictionary containing detector name, landmark support, and config.
        """
        return {
            'name': self.__class__.__name__,
            'supports_landmarks': self._supports_landmarks,
            'config': self.config,
        }

    def __call__(self, image: np.ndarray, **kwargs: Any) -> list[Face]:
        """Callable shortcut for the `detect` method.

        Args:
            image: Input image as numpy array with shape (H, W, C) in BGR format.
            **kwargs: Additional detection parameters.

        Returns:
            List of detected Face objects.
        """
        return self.detect(image, **kwargs)
