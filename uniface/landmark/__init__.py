# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from .base import BaseLandmarker
from .models import Landmark106


def create_landmarker(method: str = '2d106det', **kwargs) -> BaseLandmarker:
    """
    Factory function to create facial landmark predictors.

    Args:
        method (str): Landmark prediction method. Options: '106'.
        **kwargs: Model-specific parameters.

    Returns:
        Initialized landmarker instance.
    """
    method = method.lower()
    if method == '2d106det':
        return Landmark106(**kwargs)
    else:
        available = ['2d106det']
        raise ValueError(f"Unsupported method: '{method}'. Available: {available}")


__all__ = ['BaseLandmarker', 'Landmark106', 'create_landmarker']
