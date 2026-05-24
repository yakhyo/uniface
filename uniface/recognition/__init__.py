# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


from typing_extensions import deprecated

from .adaface import AdaFace
from .arcface import ArcFace
from .base import BaseRecognizer
from .edgeface import EdgeFace
from .mobileface import MobileFace
from .sphereface import SphereFace


@deprecated(
    'create_recognizer() is deprecated and will be removed in uniface 4.0. '
    'Instantiate the recognizer class directly, e.g. '
    '`from uniface.recognition import ArcFace; ArcFace(**kwargs)`.'
)
def create_recognizer(method: str = 'arcface', **kwargs) -> BaseRecognizer:
    """
    Factory function to create face recognizers.

    .. deprecated:: 3.7.0
        Use the recognizer class directly (``ArcFace``, ``AdaFace``,
        ``EdgeFace``, ``MobileFace``, ``SphereFace``). This factory will be
        removed in uniface 4.0.

    This function initializes and returns a face recognizer instance based on the
    specified method. It acts as a high-level interface to the underlying
    model classes like ArcFace, AdaFace, MobileFace, etc.

    Args:
        method (str): The recognition method to use.
            Options: 'arcface' (default), 'adaface', 'edgeface', 'mobileface', 'sphereface'.
        **kwargs: Model-specific parameters passed to the recognizer's constructor.
            For example, `model_name` can be used to select a specific
            pre-trained weight from the available enums (e.g., `ArcFaceWeights.MNET`).

    Returns:
        BaseRecognizer: An initialized recognizer instance ready for use.

    Raises:
        ValueError: If the specified `method` is not supported.

    Examples:
        >>> # Create the default ArcFace recognizer
        >>> from uniface.recognition import ArcFace
        >>> recognizer = ArcFace()

        >>> # Create an AdaFace recognizer
        >>> from uniface.recognition import AdaFace
        >>> from uniface.constants import AdaFaceWeights
        >>> recognizer = AdaFace(model_name=AdaFaceWeights.IR_101)

        >>> # Create a specific MobileFace recognizer
        >>> from uniface.recognition import MobileFace
        >>> from uniface.constants import MobileFaceWeights
        >>> recognizer = MobileFace(model_name=MobileFaceWeights.MNET_V2)

        >>> # Create a SphereFace recognizer
        >>> from uniface.recognition import SphereFace
        >>> recognizer = SphereFace()

        >>> # Create an EdgeFace recognizer
        >>> from uniface.recognition import EdgeFace
        >>> from uniface.constants import EdgeFaceWeights
        >>> recognizer = EdgeFace(model_name=EdgeFaceWeights.XXS)
    """
    method = method.lower()

    if method == 'arcface':
        return ArcFace(**kwargs)
    elif method == 'adaface':
        return AdaFace(**kwargs)
    elif method == 'edgeface':
        return EdgeFace(**kwargs)
    elif method == 'mobileface':
        return MobileFace(**kwargs)
    elif method == 'sphereface':
        return SphereFace(**kwargs)
    else:
        available = ['arcface', 'adaface', 'edgeface', 'mobileface', 'sphereface']
        raise ValueError(f"Unsupported method: '{method}'. Available: {available}")


__all__ = ['AdaFace', 'ArcFace', 'BaseRecognizer', 'EdgeFace', 'MobileFace', 'SphereFace', 'create_recognizer']
