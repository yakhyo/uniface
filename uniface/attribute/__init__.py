# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from typing import Any

import numpy as np

from uniface.attribute.age_gender import AgeGender
from uniface.attribute.base import Attribute
from uniface.attribute.fairface import FairFace
from uniface.constants import AgeGenderWeights, DDAMFNWeights, FairFaceWeights
from uniface.types import AttributeResult, EmotionResult

try:
    from uniface.attribute.emotion import Emotion

    _EMOTION_AVAILABLE = True
except ImportError:
    _EMOTION_AVAILABLE = False

    class Emotion(Attribute):  # type: ignore[no-redef]
        """Stub for Emotion when PyTorch is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("Emotion requires optional dependency 'torch'. Install with: pip install torch")

        def _initialize_model(self) -> None: ...
        def preprocess(self, image: np.ndarray, *args: Any) -> Any: ...
        def postprocess(self, prediction: Any) -> Any: ...
        def predict(self, image: np.ndarray, *args: Any) -> Any: ...


__all__ = [
    'AgeGender',
    'AttributeResult',
    'Emotion',
    'EmotionResult',
    'FairFace',
    'create_attribute_predictor',
]

_ATTRIBUTE_MODELS = {
    **dict.fromkeys(AgeGenderWeights, AgeGender),
    **dict.fromkeys(FairFaceWeights, FairFace),
}

if _EMOTION_AVAILABLE:
    _ATTRIBUTE_MODELS.update(dict.fromkeys(DDAMFNWeights, Emotion))


def create_attribute_predictor(
    model_name: AgeGenderWeights | DDAMFNWeights | FairFaceWeights, **kwargs: Any
) -> Attribute:
    """Factory function to create an attribute predictor instance.

    Args:
        model_name: The enum corresponding to the desired attribute model
            (e.g., AgeGenderWeights.DEFAULT, DDAMFNWeights.AFFECNET7,
            or FairFaceWeights.DEFAULT).
        **kwargs: Additional keyword arguments passed to the model constructor.

    Returns:
        An initialized Attribute predictor (AgeGender, FairFace, or Emotion).

    Raises:
        ValueError: If the provided model_name is not a supported enum.
    """
    model_class = _ATTRIBUTE_MODELS.get(model_name)

    if model_class is None:
        raise ValueError(
            f'Unsupported attribute model: {model_name}. '
            f'Please choose from AgeGenderWeights, FairFaceWeights, or DDAMFNWeights.'
        )

    return model_class(model_name=model_name, **kwargs)
