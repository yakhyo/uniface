# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


from .base import BaseRecognizer
from .models import ArcFace, MobileFace, SphereFace


def create_recognizer(method: str = 'arcface', **kwargs) -> BaseRecognizer:
    """
    Factory function to create face recognizers.

    This function initializes and returns a face recognizer instance based on the
    specified method. It acts as a high-level interface to the underlying
    model classes like ArcFace, MobileFace, etc.

    Args:
        method (str): The recognition method to use.
            Options: 'arcface' (default), 'mobileface', 'sphereface'.
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

        >>> # Create a specific MobileFace recognizer
        >>> from uniface.constants import MobileFaceWeights
        >>> recognizer = create_recognizer('mobileface', model_name=MobileFaceWeights.MNET_V2)

        >>> # Create a SphereFace recognizer
        >>> recognizer = create_recognizer('sphereface')
    """
    method = method.lower()

    if method == 'arcface':
        return ArcFace(**kwargs)
    elif method == 'mobileface':
        return MobileFace(**kwargs)
    elif method == 'sphereface':
        return SphereFace(**kwargs)
    else:
        available = ['arcface', 'mobileface', 'sphereface']
        raise ValueError(f"Unsupported method: '{method}'. Available: {available}")


__all__ = ['ArcFace', 'BaseRecognizer', 'MobileFace', 'SphereFace', 'create_recognizer']
