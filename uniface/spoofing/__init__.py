# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from typing import Optional

from uniface.constants import MiniFASNetWeights

from .base import BaseSpoofer
from .minifasnet import MiniFASNet

__all__ = [
    'BaseSpoofer',
    'MiniFASNet',
    'MiniFASNetWeights',
    'create_spoofer',
]


def create_spoofer(
    model_name: MiniFASNetWeights = MiniFASNetWeights.V2,
    scale: Optional[float] = None,
) -> MiniFASNet:
    """
    Factory function to create a face anti-spoofing model.

    This is a convenience function that creates a MiniFASNet instance
    with the specified model variant and optional custom scale.

    Args:
        model_name (MiniFASNetWeights): The model variant to use.
            Options:
                - MiniFASNetWeights.V2: Improved version (default), uses scale=2.7
                - MiniFASNetWeights.V1SE: Squeeze-and-excitation version, uses scale=4.0
            Defaults to MiniFASNetWeights.V2.
        scale (Optional[float]): Custom crop scale factor for face region.
            If None, uses the default scale for the selected model variant.

    Returns:
        MiniFASNet: An initialized face anti-spoofing model.

    Example:
        >>> from uniface.spoofing import create_spoofer, MiniFASNetWeights
        >>> from uniface import RetinaFace
        >>>
        >>> # Create with default settings (V2 model)
        >>> spoofer = create_spoofer()
        >>>
        >>> # Create with V1SE model
        >>> spoofer = create_spoofer(model_name=MiniFASNetWeights.V1SE)
        >>>
        >>> # Create with custom scale
        >>> spoofer = create_spoofer(scale=3.0)
        >>>
        >>> # Use with face detector
        >>> detector = RetinaFace()
        >>> faces = detector.detect(image)
        >>> for face in faces:
        ...     label_idx, score = spoofer.predict(image, face['bbox'])
        ...     # label_idx: 0 = Fake, 1 = Real
        ...     label = 'Real' if label_idx == 1 else 'Fake'
        ...     print(f'{label}: {score:.2%}')
    """
    return MiniFASNet(model_name=model_name, scale=scale)
