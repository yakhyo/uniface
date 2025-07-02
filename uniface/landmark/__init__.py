# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from .models import Landmark106
from .base import BaseLandmarker


def create_landmarker(method: str = '2d106', **kwargs) -> BaseLandmarker:
    """
    Factory function to create facial landmark predictors.

    Args:
        method (str): Landmark prediction method. Options: '106'.
        **kwargs: Model-specific parameters.

    Returns:
        Initialized landmarker instance.
    """
    method = method.lower()
    if method == 'insightface_106':
        return Landmark106(**kwargs)
    else:
        available = ['insightface_106']
        raise ValueError(f"Unsupported method: '{method}'. Available: {available}")


__all__ = [
    "create_landmarker",
    "Landmark106",
    "BaseLandmarker"
]
