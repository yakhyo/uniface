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
    """Factory function to create a face parsing model.

    Args:
        model_name: Model to create. Options: ParsingWeights.RESNET18/RESNET34 (BiSeNet),
            XSegWeights.DEFAULT (XSeg, requires landmarks).
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        An instance of the requested face parsing model.

    Raises:
        ValueError: If the model_name is not recognized.

    Example:
        >>> parser = create_face_parser(ParsingWeights.RESNET18)
        >>> mask = parser.parse(face_crop)
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
