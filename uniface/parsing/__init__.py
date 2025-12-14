# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from typing import Union

from uniface.constants import ParsingWeights

from .base import BaseFaceParser
from .bisenet import BiSeNet

__all__ = ['BaseFaceParser', 'BiSeNet', 'create_face_parser']


def create_face_parser(
    model_name: Union[str, ParsingWeights] = ParsingWeights.RESNET18,
) -> BaseFaceParser:
    """
    Factory function to create a face parsing model instance.

    This function provides a convenient way to instantiate face parsing models
    without directly importing the specific model classes. It supports both
    string-based and enum-based model selection.

    Args:
        model_name (Union[str, ParsingWeights]): The face parsing model to create.
            Can be either a string or a ParsingWeights enum value.
            Available options:
            - 'parsing_resnet18' or ParsingWeights.RESNET18 (default)
            - 'parsing_resnet34' or ParsingWeights.RESNET34

    Returns:
        BaseFaceParser: An instance of the requested face parsing model.

    Raises:
        ValueError: If the model_name is not recognized.

    Examples:
        >>> # Using enum
        >>> from uniface.parsing import create_face_parser
        >>> from uniface.constants import ParsingWeights
        >>> parser = create_face_parser(ParsingWeights.RESNET18)
        >>>
        >>> # Using string
        >>> parser = create_face_parser('parsing_resnet18')
        >>>
        >>> # Parse a face image
        >>> mask = parser.parse(face_crop)
    """
    # Convert string to enum if necessary
    if isinstance(model_name, str):
        try:
            model_name = ParsingWeights(model_name)
        except ValueError as e:
            valid_models = [e.value for e in ParsingWeights]
            raise ValueError(
                f"Unknown face parsing model: '{model_name}'. Valid options are: {', '.join(valid_models)}"
            ) from e

    # All parsing models use the same BiSeNet class
    return BiSeNet(model_name=model_name)
