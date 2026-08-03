# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from typing import Any

import numpy as np

from uniface.attribute.age_gender import AgeGender
from uniface.attribute.base import BaseAttribute
from uniface.attribute.faceattribnet import FaceAttribNet
from uniface.attribute.fairface import FairFace
from uniface.types import DemographyResult, EmotionResult, Face, FaceStateResult

try:
    from uniface.attribute.emotion import Emotion
except ImportError:

    class Emotion(BaseAttribute):  # type: ignore[no-redef]
        """Stub for Emotion when PyTorch is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("Emotion requires optional dependency 'torch'. Install with: pip install torch")

        def _initialize_model(self) -> None: ...
        def preprocess(self, image: np.ndarray, *args: Any) -> Any: ...
        def postprocess(self, prediction: Any) -> Any: ...
        def predict(self, image: np.ndarray, face: Face) -> Any: ...


__all__ = [
    'AgeGender',
    'BaseAttribute',
    'DemographyResult',
    'Emotion',
    'EmotionResult',
    'FaceAttribNet',
    'FaceStateResult',
    'FairFace',
]
