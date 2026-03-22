# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from uniface.types import AttributeResult, EmotionResult, Face

__all__ = ['Attribute', 'AttributeResult', 'EmotionResult']


class Attribute(ABC):
    """Abstract base class for face attribute models.

    All attribute models (age-gender, emotion, FairFace, etc.) implement this
    interface so they can be used interchangeably inside ``FaceAnalyzer``.

    The ``predict`` method accepts an image and a :class:`Face` object.  Each
    subclass extracts what it needs (bbox, landmarks) from the Face, runs
    inference, writes the results back to the Face **and** returns a typed
    result dataclass.
    """

    @abstractmethod
    def _initialize_model(self) -> None:
        """Load model weights and create the inference session."""
        raise NotImplementedError('Subclasses must implement the _initialize_model method.')

    @abstractmethod
    def preprocess(self, image: np.ndarray, *args: Any) -> Any:
        """Preprocess the input data for the model.

        Args:
            image: The input image in BGR format.
            *args: Subclass-specific data (bbox, landmarks, etc.).

        Returns:
            Preprocessed data ready for model inference.
        """
        raise NotImplementedError('Subclasses must implement the preprocess method.')

    @abstractmethod
    def postprocess(self, prediction: Any) -> Any:
        """Convert raw model output into a typed result dataclass.

        Args:
            prediction: Raw output from the model.

        Returns:
            An ``AttributeResult`` or ``EmotionResult``.
        """
        raise NotImplementedError('Subclasses must implement the postprocess method.')

    @abstractmethod
    def predict(self, image: np.ndarray, face: Face) -> AttributeResult | EmotionResult:
        """Run end-to-end prediction and enrich the Face in-place.

        Each subclass extracts what it needs from *face* (e.g. ``face.bbox``
        or ``face.landmarks``), runs the full preprocess-infer-postprocess
        pipeline, writes relevant fields back to *face*, and returns the
        result dataclass.

        Args:
            image: The full input image in BGR format.
            face: Detected face whose attribute fields will be populated.

        Returns:
            The prediction result (``AttributeResult`` or ``EmotionResult``).
        """
        raise NotImplementedError('Subclasses must implement the predict method.')

    def __call__(self, image: np.ndarray, face: Face) -> AttributeResult | EmotionResult:
        """Callable shortcut for :meth:`predict`."""
        return self.predict(image, face)
