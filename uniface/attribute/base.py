# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

__all__ = ['Attribute', 'AttributeResult']


@dataclass(slots=True)
class AttributeResult:
    """
    Unified result structure for face attribute prediction.

    This dataclass provides a consistent return type across different attribute
    prediction models (e.g., AgeGender, FairFace), enabling interoperability
    and unified handling of results.

    Attributes:
        gender: Predicted gender (0=Female, 1=Male).
        age: Exact age in years. Provided by AgeGender model, None for FairFace.
        age_group: Age range string like "20-29". Provided by FairFace, None for AgeGender.
        race: Race/ethnicity label. Provided by FairFace only.

    Properties:
        sex: Gender as a human-readable string ("Female" or "Male").

    Examples:
        >>> # AgeGender result
        >>> result = AttributeResult(gender=1, age=25)
        >>> result.sex
        'Male'
        >>> result.age
        25

        >>> # FairFace result
        >>> result = AttributeResult(gender=0, age_group="20-29", race="East Asian")
        >>> result.sex
        'Female'
        >>> result.race
        'East Asian'
    """

    gender: int
    age: Optional[int] = None
    age_group: Optional[str] = None
    race: Optional[str] = None

    @property
    def sex(self) -> str:
        """Get gender as a string label (Female or Male)."""
        return 'Female' if self.gender == 0 else 'Male'


class Attribute(ABC):
    """
    Abstract base class for face attribute models.

    This class defines the common interface that all attribute models
    (e.g., age-gender, emotion) must implement. It ensures a consistent API
    across different attribute prediction modules in the library, making them
    interchangeable and easy to use.
    """

    @abstractmethod
    def _initialize_model(self) -> None:
        """
        Initializes the underlying model for inference.

        This method should handle loading model weights, creating the
        inference session (e.g., ONNX Runtime, PyTorch), and any necessary
        warm-up procedures to prepare the model for prediction.
        """
        raise NotImplementedError('Subclasses must implement the _initialize_model method.')

    @abstractmethod
    def preprocess(self, image: np.ndarray, *args: Any) -> Any:
        """
        Preprocesses the input data for the model.

        This method should take a raw image and any other necessary data
        (like bounding boxes or landmarks) and convert it into the format
        expected by the model's inference engine (e.g., a blob or tensor).

        Args:
            image (np.ndarray): The input image containing the face, typically
                                in BGR format.
            *args: Additional arguments required for preprocessing, such as
                   bounding boxes or facial landmarks.

        Returns:
            The preprocessed data ready for model inference.
        """
        raise NotImplementedError('Subclasses must implement the preprocess method.')

    @abstractmethod
    def postprocess(self, prediction: Any) -> Any:
        """
        Postprocesses the raw model output into a human-readable format.

        This method takes the raw output from the model's inference and
        converts it into a meaningful result, such as an age value, a gender
        label, or an emotion category.

        Args:
            prediction (Any): The raw output from the model's inference.

        Returns:
            The final, processed attributes.
        """
        raise NotImplementedError('Subclasses must implement the postprocess method.')

    @abstractmethod
    def predict(self, image: np.ndarray, *args: Any) -> Any:
        """
        Performs end-to-end attribute prediction on a given image.

        This method orchestrates the full pipeline: it calls the preprocess,
        inference, and postprocess steps to return the final, user-friendly
        attribute prediction.

        Args:
            image (np.ndarray): The input image containing the face.
            *args: Additional data required for prediction, such as a bounding
                   box or landmarks.

        Returns:
            The final predicted attributes.
        """
        raise NotImplementedError('Subclasses must implement the predict method.')

    def __call__(self, *args, **kwargs) -> Any:
        """
        Provides a convenient, callable shortcut for the `predict` method.
        """
        return self.predict(*args, **kwargs)
