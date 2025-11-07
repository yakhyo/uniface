# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""
Base classes for face detection.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any


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
    def detect(self, image: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect faces in an image.

        Args:
            image (np.ndarray): Input image as numpy array with shape (H, W, C)
            **kwargs: Additional detection parameters

        Returns:
            Tuple[np.ndarray, np.ndarray]: (detections, landmarks)
                - detections: Bounding boxes with confidence scores, shape (N, 5)
                  Format: [x_min, y_min, x_max, y_max, confidence]
                - landmarks: Facial landmark points, shape (N, 5, 2) for 5-point landmarks
                  or (N, 68, 2) for 68-point landmarks. Empty array if not supported.
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
    def postprocess(self, outputs, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Postprocess model outputs to get final detections.

        Args:
            outputs: Raw model outputs
            **kwargs: Additional postprocessing parameters

        Returns:
            Tuple[np.ndarray, np.ndarray]: (detections, landmarks)
        """
        pass

    def __str__(self) -> str:
        """String representation of the detector."""
        return f"{self.__class__.__name__}({self.config})"

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
            'config': self.config
        }
