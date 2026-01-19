# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from uniface.constants import ParsingWeights, XSegWeights

from .base import BaseFaceParser
from .bisenet import BiSeNet
from .xseg import XSeg

__all__ = ['BaseFaceParser', 'BiSeNet', 'XSeg', 'create_face_parser']


def create_face_parser(
    model_name: str | ParsingWeights | XSegWeights = ParsingWeights.RESNET18,
    **kwargs,
) -> BaseFaceParser:
    """Factory function to create a face parsing model instance.

    This function provides a convenient way to instantiate face parsing models
    without directly importing the specific model classes.

    Args:
        model_name: The face parsing model to create. Can be either a string
            or an enum value. Available options:
            - 'parsing_resnet18' or ParsingWeights.RESNET18 (default) - BiSeNet
            - 'parsing_resnet34' or ParsingWeights.RESNET34 - BiSeNet
            - 'xseg' or XSegWeights.DEFAULT - XSeg (requires landmarks)
        **kwargs: Additional arguments passed to the model constructor.
            For XSeg: align_size (int), blur_sigma (float), providers (list).

    Returns:
        An instance of the requested face parsing model.

    Raises:
        ValueError: If the model_name is not recognized.

    Example:
        >>> from uniface.parsing import create_face_parser
        >>> from uniface.constants import ParsingWeights, XSegWeights
        >>> # BiSeNet parser
        >>> parser = create_face_parser(ParsingWeights.RESNET18)
        >>> mask = parser.parse(face_crop)
        >>> # XSeg parser (requires landmarks)
        >>> xseg = create_face_parser(XSegWeights.DEFAULT, blur_sigma=5)
        >>> mask = xseg.parse(image, landmarks)
    """
    # Handle XSegWeights
    if isinstance(model_name, XSegWeights):
        return XSeg(model_name=model_name, **kwargs)

    # Convert string to enum if necessary
    if isinstance(model_name, str):
        # Try XSegWeights first
        try:
            xseg_model = XSegWeights(model_name)
            return XSeg(model_name=xseg_model, **kwargs)
        except ValueError:
            pass

        # Try ParsingWeights
        try:
            model_name = ParsingWeights(model_name)
        except ValueError as e:
            valid_parsing = [m.value for m in ParsingWeights]
            valid_xseg = [m.value for m in XSegWeights]
            valid_models = valid_parsing + valid_xseg
            raise ValueError(
                f"Unknown face parsing model: '{model_name}'. Valid options are: {', '.join(valid_models)}"
            ) from e

    # BiSeNet models
    return BiSeNet(model_name=model_name, **kwargs)
