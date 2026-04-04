# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


from .adaface import AdaFace
from .arcface import ArcFace
from .base import BaseRecognizer
from .edgeface import EdgeFace
from .mobileface import MobileFace
from .sphereface import SphereFace


def create_recognizer(method: str = 'arcface', **kwargs) -> BaseRecognizer:
    """
    Factory function to create face recognizers.

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
        >>> recognizer = create_recognizer()

        >>> # Create an AdaFace recognizer
        >>> from uniface.constants import AdaFaceWeights
        >>> recognizer = create_recognizer('adaface', model_name=AdaFaceWeights.IR_101)

        >>> # Create a specific MobileFace recognizer
        >>> from uniface.constants import MobileFaceWeights
        >>> recognizer = create_recognizer('mobileface', model_name=MobileFaceWeights.MNET_V2)

        >>> # Create a SphereFace recognizer
        >>> recognizer = create_recognizer('sphereface')

        >>> # Create an EdgeFace recognizer
        >>> from uniface.constants import EdgeFaceWeights
        >>> recognizer = create_recognizer('edgeface', model_name=EdgeFaceWeights.XXS)
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
