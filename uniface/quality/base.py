# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from uniface.types import QualityResult

__all__ = ['BaseQualityEstimator', 'QualityResult']


class BaseQualityEstimator(ABC):
    """
    Abstract base class for face image quality assessment models.

    Quality estimators predict a single scalar score from an aligned face crop,
    where higher values indicate better quality (sharpness, frontalness,
    illumination, low occlusion). The score is typically used to filter or
    rank faces prior to recognition.
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
    def preprocess(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Preprocess an aligned face crop for model inference.

        Args:
            aligned_face: An aligned face crop in BGR format, sized to the
                model's expected input (typically 112x112).

        Returns:
            The preprocessed image tensor ready for inference,
            typically with shape (1, C, H, W).
        """
        raise NotImplementedError('Subclasses must implement the preprocess method.')

    @abstractmethod
    def score_aligned(self, aligned_face: np.ndarray) -> QualityResult:
        """
        Score a pre-aligned face crop.

        Use this when the input is already aligned to the model's expected
        layout (e.g., from `uniface.face_alignment`).

        Args:
            aligned_face: Aligned face crop in BGR format.

        Returns:
            QualityResult with the predicted score.
        """
        raise NotImplementedError('Subclasses must implement the score_aligned method.')

    @abstractmethod
    def predict(self, image: np.ndarray, landmarks: np.ndarray) -> QualityResult:
        """
        Perform end-to-end quality estimation from a full image and landmarks.

        Aligns the face using the provided 5-point landmarks, then scores it.

        Args:
            image: Input image in BGR format containing the face.
            landmarks: (5, 2) array of 5-point facial landmarks for alignment.

        Returns:
            QualityResult with the predicted score.
        """
        raise NotImplementedError('Subclasses must implement the predict method.')

    def __call__(self, image: np.ndarray, landmarks: np.ndarray) -> QualityResult:
        """Callable shortcut for :meth:`predict`."""
        return self.predict(image, landmarks)
