# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from typing import Any

import numpy as np

from uniface.attribute.age_gender import AgeGender
from uniface.attribute.base import Attribute
from uniface.attribute.fairface import FairFace
from uniface.constants import AgeGenderWeights, DDAMFNWeights, FairFaceWeights
from uniface.types import AttributeResult, EmotionResult, Face

# Emotion requires PyTorch - make it optional
try:
    from uniface.attribute.emotion import Emotion

    _EMOTION_AVAILABLE = True
except ImportError:
    Emotion = None
    _EMOTION_AVAILABLE = False

# Public API for the attribute module
__all__ = [
    'AgeGender',
    'AttributeResult',
    'Emotion',
    'EmotionResult',
    'FairFace',
    'create_attribute_predictor',
    'predict_attributes',
]

# A mapping from model enums to their corresponding attribute classes
_ATTRIBUTE_MODELS = {
    **dict.fromkeys(AgeGenderWeights, AgeGender),
    **dict.fromkeys(FairFaceWeights, FairFace),
}

# Add Emotion models only if PyTorch is available
if _EMOTION_AVAILABLE:
    _ATTRIBUTE_MODELS.update(dict.fromkeys(DDAMFNWeights, Emotion))


def create_attribute_predictor(
    model_name: AgeGenderWeights | DDAMFNWeights | FairFaceWeights, **kwargs: Any
) -> Attribute:
    """
    Factory function to create an attribute predictor instance.

    This high-level API simplifies the creation of attribute models by
    dynamically selecting the correct class based on the provided model enum.

    Args:
        model_name: The enum corresponding to the desired attribute model
                    (e.g., AgeGenderWeights.DEFAULT, DDAMFNWeights.AFFECNET7,
                    or FairFaceWeights.DEFAULT).
        **kwargs: Additional keyword arguments to pass to the model's constructor.

    Returns:
        An initialized instance of an Attribute predictor class
        (e.g., AgeGender, FairFace, or Emotion).

    Raises:
        ValueError: If the provided model_name is not a supported enum.
    """
    model_class = _ATTRIBUTE_MODELS.get(model_name)

    if model_class is None:
        raise ValueError(
            f'Unsupported attribute model: {model_name}. '
            f'Please choose from AgeGenderWeights, FairFaceWeights, or DDAMFNWeights.'
        )

    # Pass model_name to the constructor, as some classes might need it
    return model_class(model_name=model_name, **kwargs)


def predict_attributes(image: np.ndarray, faces: list[Face], predictor: Attribute) -> list[Face]:
    """
    High-level API to predict attributes for multiple detected faces.

    This function iterates through a list of Face objects, runs the
    specified attribute predictor on each one, and updates the Face
    objects with the predicted attributes.

    Args:
        image (np.ndarray): The full input image in BGR format.
        faces (List[Face]): A list of Face objects from face detection.
        predictor (Attribute): An initialized attribute predictor instance,
                               created by `create_attribute_predictor`.

    Returns:
        List[Face]: The list of Face objects with updated attribute fields.
    """
    for face in faces:
        if isinstance(predictor, AgeGender):
            result = predictor(image, face.bbox)
            face.gender = result.gender
            face.age = result.age
        elif isinstance(predictor, FairFace):
            result = predictor(image, face.bbox)
            face.gender = result.gender
            face.age_group = result.age_group
            face.race = result.race
        elif isinstance(predictor, Emotion):
            result = predictor(image, face.landmarks)
            face.emotion = result.emotion
            face.emotion_confidence = result.confidence

    return faces
