# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from typing_extensions import deprecated

from uniface.constants import ParsingWeights, XSegWeights

from .base import BaseFaceParser
from .bisenet import BiSeNet
from .xseg import XSeg

__all__ = ['BaseFaceParser', 'BiSeNet', 'XSeg', 'create_face_parser']


@deprecated(
    'create_face_parser() is deprecated and will be removed in uniface 4.0. '
    'Instantiate the parser class directly, e.g. '
    '`from uniface.parsing import BiSeNet; BiSeNet(model_name=...)`.'
)
def create_face_parser(
    model_name: str | ParsingWeights | XSegWeights = ParsingWeights.RESNET18,
    **kwargs,
) -> BaseFaceParser:
    """Factory function to create a face parsing model.

    .. deprecated:: 3.7.0
        Use the parser class directly (``BiSeNet``, ``XSeg``). This factory
        will be removed in uniface 4.0.

    Args:
        model_name: Model to create. Options: ParsingWeights.RESNET18/RESNET34 (BiSeNet),
            XSegWeights.DEFAULT (XSeg, requires landmarks).
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        An instance of the requested face parsing model.

    Raises:
        ValueError: If the model_name is not recognized.

    Example:
        >>> from uniface.parsing import BiSeNet
        >>> parser = BiSeNet(model_name=ParsingWeights.RESNET18)
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
