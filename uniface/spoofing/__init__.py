# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from uniface.constants import MiniFASNetWeights
from uniface.types import SpoofingResult

from .base import BaseSpoofer
from .minifasnet import MiniFASNet

__all__ = [
    'BaseSpoofer',
    'MiniFASNet',
    'MiniFASNetWeights',
    'SpoofingResult',
    'create_spoofer',
]


def create_spoofer(
    model_name: MiniFASNetWeights = MiniFASNetWeights.V2,
    scale: float | None = None,
) -> MiniFASNet:
    """Factory function to create a face anti-spoofing model.

    This is a convenience function that creates a MiniFASNet instance
    with the specified model variant and optional custom scale.

    Args:
        model_name: The model variant to use. Options:
            - MiniFASNetWeights.V2: Improved version (default), uses scale=2.7
            - MiniFASNetWeights.V1SE: Squeeze-and-excitation version, uses scale=4.0
        scale: Custom crop scale factor for face region. If None, uses the
            default scale for the selected model variant.

    Returns:
        An initialized face anti-spoofing model.

    Example:
        >>> from uniface.spoofing import create_spoofer, MiniFASNetWeights
        >>> spoofer = create_spoofer()
        >>> result = spoofer.predict(image, face.bbox)
        >>> print(f'Is real: {result.is_real}, Confidence: {result.confidence:.2%}')
    """
    return MiniFASNet(model_name=model_name, scale=scale)
