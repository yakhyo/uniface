# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class BaseDetector(ABC):
    """
    Abstract base class for all face detectors.

    This class defines the interface that all face detectors must implement,
    ensuring consistency across different detection methods.
    """

    def __init__(self, **kwargs):
        """Initialize the detector with configuration parameters."""
        self.config = kwargs

    @abstractmethod
    def detect(self, image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """
        Detect faces in an image.

        Args:
            image (np.ndarray): Input image as numpy array with shape (H, W, C)
            **kwargs: Additional detection parameters

        Returns:
            List[Dict[str, Any]]: List of detected faces, where each dictionary contains:
                - 'bbox' (np.ndarray): Bounding box coordinates with shape (4,) as [x1, y1, x2, y2]
                - 'confidence' (float): Detection confidence score (0.0 to 1.0)
                - 'landmarks' (np.ndarray): Facial landmarks with shape (5, 2) for 5-point landmarks
                  or (68, 2) for 68-point landmarks. Empty array if not supported.

        Example:
            >>> faces = detector.detect(image)
            >>> for face in faces:
            ...     bbox = face['bbox']  # np.ndarray with shape (4,)
            ...     confidence = face['confidence']  # float
            ...     landmarks = face['landmarks']  # np.ndarray with shape (5, 2)
        """
        pass

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess input image for detection.

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Preprocessed image tensor
        """
        pass

    @abstractmethod
    def postprocess(self, outputs, **kwargs) -> Any:
        """
        Postprocess model outputs to get final detections.

        Args:
            outputs: Raw model outputs
            **kwargs: Additional postprocessing parameters

        Returns:
            Any: Processed outputs (implementation-specific format, typically tuple of arrays)
        """
        pass

    def __str__(self) -> str:
        """String representation of the detector."""
        return f'{self.__class__.__name__}({self.config})'

    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()

    @property
    def supports_landmarks(self) -> bool:
        """
        Whether this detector supports landmark detection.

        Returns:
            bool: True if landmarks are supported, False otherwise
        """
        return hasattr(self, '_supports_landmarks') and self._supports_landmarks

    def get_info(self) -> Dict[str, Any]:
        """
        Get detector information and configuration.

        Returns:
            Dict[str, Any]: Detector information
        """
        return {
            'name': self.__class__.__name__,
            'supports_landmarks': self._supports_landmarks,
            'config': self.config,
        }
