# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from abc import ABC, abstractmethod

import numpy as np


class BaseLandmarker(ABC):
    """
    Abstract Base Class for all facial landmark models.
    """

    @abstractmethod
    def get_landmarks(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Predicts facial landmarks for a given face bounding box.

        This method defines the standard interface for all landmark predictors.
        It takes a full image and a bounding box for a single face and returns
        the predicted keypoints for that face.

        Args:
            image (np.ndarray): The full source image in BGR format.
            bbox (np.ndarray): A bounding box of a face [x1, y1, x2, y2].

        Returns:
            np.ndarray: An array of predicted landmark points with shape (N, 2),
                        where N is the number of landmarks.
        """
        raise NotImplementedError

    def __call__(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Callable shortcut for the `get_landmarks` method.

        Args:
            image (np.ndarray): The full source image in BGR format.
            bbox (np.ndarray): A bounding box of a face [x1, y1, x2, y2].

        Returns:
            np.ndarray: An array of predicted landmark points with shape (N, 2).
        """
        return self.get_landmarks(image, bbox)
