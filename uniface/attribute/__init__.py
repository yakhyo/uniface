# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from typing import Any, Dict, List, Union

import numpy as np

from uniface.attribute.age_gender import AgeGender
from uniface.attribute.base import Attribute
from uniface.constants import AgeGenderWeights, DDAMFNWeights

# Emotion requires PyTorch - make it optional
try:
    from uniface.attribute.emotion import Emotion

    _EMOTION_AVAILABLE = True
except ImportError:
    Emotion = None
    _EMOTION_AVAILABLE = False

# Public API for the attribute module
__all__ = ['AgeGender', 'Emotion', 'create_attribute_predictor', 'predict_attributes']

# A mapping from model enums to their corresponding attribute classes
_ATTRIBUTE_MODELS = {
    **{model: AgeGender for model in AgeGenderWeights},
}

# Add Emotion models only if PyTorch is available
if _EMOTION_AVAILABLE:
    _ATTRIBUTE_MODELS.update({model: Emotion for model in DDAMFNWeights})


def create_attribute_predictor(model_name: Union[AgeGenderWeights, DDAMFNWeights], **kwargs: Any) -> Attribute:
    """
    Factory function to create an attribute predictor instance.

    This high-level API simplifies the creation of attribute models by
    dynamically selecting the correct class based on the provided model enum.

    Args:
        model_name: The enum corresponding to the desired attribute model
                    (e.g., AgeGenderWeights.DEFAULT or DDAMFNWeights.AFFECNET7).
        **kwargs: Additional keyword arguments to pass to the model's constructor.

    Returns:
        An initialized instance of an Attribute predictor class (e.g., AgeGender).

    Raises:
        ValueError: If the provided model_name is not a supported enum.
    """
    model_class = _ATTRIBUTE_MODELS.get(model_name)

    if model_class is None:
        raise ValueError(
            f'Unsupported attribute model: {model_name}. Please choose from AgeGenderWeights or DDAMFNWeights.'
        )

    # Pass model_name to the constructor, as some classes might need it
    return model_class(model_name=model_name, **kwargs)


def predict_attributes(
    image: np.ndarray, detections: List[Dict[str, np.ndarray]], predictor: Attribute
) -> List[Dict[str, Any]]:
    """
    High-level API to predict attributes for multiple detected faces.

    This function iterates through a list of face detections, runs the
    specified attribute predictor on each one, and appends the results back
    into the detection dictionary.

    Args:
        image (np.ndarray): The full input image in BGR format.
        detections (List[Dict]): A list of detection results, where each dict
                                 must contain a 'bbox' and optionally 'landmark'.
        predictor (Attribute): An initialized attribute predictor instance,
                               created by `create_attribute_predictor`.

    Returns:
        The list of detections, where each dictionary is updated with a new
        'attributes' key containing the prediction result.
    """
    for face in detections:
        # Initialize attributes dict if it doesn't exist
        if 'attributes' not in face:
            face['attributes'] = {}

        if isinstance(predictor, AgeGender):
            gender_id, age = predictor(image, face['bbox'])
            face['attributes']['gender_id'] = gender_id
            face['attributes']['age'] = age
        elif isinstance(predictor, Emotion):
            emotion, confidence = predictor(image, face['landmark'])
            face['attributes']['emotion'] = emotion
            face['attributes']['confidence'] = confidence

    return detections
