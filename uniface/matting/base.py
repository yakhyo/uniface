# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseMatting(ABC):
    """Abstract base class for portrait matting models.

    Unlike face parsers that operate on face crops and produce class labels or
    face-region masks, matting models operate on full images and produce a soft
    alpha matte (float32 in [0, 1]) separating foreground from background.

    Subclasses must implement the full pipeline: model initialization,
    preprocessing, postprocessing, and the ``predict`` entry point.
    """

    @abstractmethod
    def _initialize_model(self) -> None:
        """Initialize the underlying model for inference.

        This method should handle loading model weights, creating the
        inference session (e.g., ONNX Runtime), and any necessary
        setup procedures to prepare the model for prediction.

        Raises:
            RuntimeError: If the model fails to load or initialize.
        """
        raise NotImplementedError('Subclasses must implement the _initialize_model method.')

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Preprocess the input image for model inference.

        Args:
            image: An image in BGR format with shape ``(H, W, 3)``.

        Returns:
            A tuple of ``(tensor, orig_h, orig_w)`` where *tensor* is the
            preprocessed image ready for inference.
        """
        raise NotImplementedError('Subclasses must implement the preprocess method.')

    @abstractmethod
    def postprocess(self, outputs: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
        """Postprocess raw model outputs into an alpha matte.

        Args:
            outputs: Raw outputs from the model inference.
            original_size: Original image size as ``(width, height)``.

        Returns:
            Alpha matte with shape ``(H, W)`` and values in ``[0, 1]``.
        """
        raise NotImplementedError('Subclasses must implement the postprocess method.')

    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run end-to-end matting on an image.

        Args:
            image: An image in BGR format with shape ``(H, W, 3)``.

        Returns:
            Alpha matte with shape ``(H, W)``, float32 in ``[0, 1]``.

        Example:
            >>> matting = create_matting_model()
            >>> matte = matting.predict(image)
            >>> print(f'Matte shape: {matte.shape}, dtype: {matte.dtype}')
        """
        raise NotImplementedError('Subclasses must implement the predict method.')

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Callable shortcut for :meth:`predict`.

        Args:
            image: An image in BGR format with shape ``(H, W, 3)``.

        Returns:
            Alpha matte with shape ``(H, W)``, float32 in ``[0, 1]``.
        """
        return self.predict(image)
