# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from typing_extensions import deprecated

from .base import BaseLandmarker
from .models import Landmark106
from .pipnet import PIPNet


@deprecated(
    'create_landmarker() is deprecated and will be removed in uniface 4.0. '
    'Instantiate the landmarker class directly, e.g. '
    '`from uniface.landmark import Landmark106; Landmark106(**kwargs)`.'
)
def create_landmarker(method: str = '2d106det', **kwargs) -> BaseLandmarker:
    """Factory function to create facial landmark predictors.

    .. deprecated:: 3.7.0
        Use the landmarker class directly (``Landmark106``, ``PIPNet``).
        This factory will be removed in uniface 4.0.

    Args:
        method (str): Landmark prediction method.
            Options:
                - ``'2d106det'`` (default): InsightFace 2d106det 106-point model.
                - ``'pipnet'``: PIPNet 98-point (WFLW) or 68-point (300W+CelebA)
                  model. Pass ``model_name=PIPNetWeights.DW300_CELEBA_68`` for
                  the 68-point variant.
        **kwargs: Model-specific parameters forwarded to the underlying class.

    Returns:
        Initialized landmarker instance.

    Raises:
        ValueError: If ``method`` is not supported.
    """
    method = method.lower()
    if method == '2d106det':
        return Landmark106(**kwargs)
    if method == 'pipnet':
        return PIPNet(**kwargs)

    available = ['2d106det', 'pipnet']
    raise ValueError(f"Unsupported method: '{method}'. Available: {available}")


__all__ = ['BaseLandmarker', 'Landmark106', 'PIPNet', 'create_landmarker']
