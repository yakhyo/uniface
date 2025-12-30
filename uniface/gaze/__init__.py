# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from uniface.types import GazeResult

from .base import BaseGazeEstimator
from .models import MobileGaze


def create_gaze_estimator(method: str = 'mobilegaze', **kwargs) -> BaseGazeEstimator:
    """
    Factory function to create gaze estimators.

    This function initializes and returns a gaze estimator instance based on the
    specified method. It acts as a high-level interface to the underlying
    model classes.

    Args:
        method (str): The gaze estimation method to use.
            Options: 'mobilegaze' (default).
        **kwargs: Model-specific parameters passed to the estimator's constructor.
            For example, `model_name` can be used to select a specific
            backbone from `GazeWeights` enum (RESNET18, RESNET34, RESNET50,
            MOBILENET_V2, MOBILEONE_S0).

    Returns:
        BaseGazeEstimator: An initialized gaze estimator instance ready for use.

    Raises:
        ValueError: If the specified `method` is not supported.

    Examples:
        >>> # Create the default MobileGaze estimator (ResNet18 backbone)
        >>> estimator = create_gaze_estimator()

        >>> # Create with MobileNetV2 backbone
        >>> from uniface.constants import GazeWeights
        >>> estimator = create_gaze_estimator('mobilegaze', model_name=GazeWeights.MOBILENET_V2)

        >>> # Use the estimator
        >>> result = estimator.estimate(face_crop)
        >>> print(f'Pitch: {result.pitch}, Yaw: {result.yaw}')
    """
    method = method.lower()

    if method in ('mobilegaze', 'mobile_gaze', 'gaze'):
        return MobileGaze(**kwargs)
    else:
        available = ['mobilegaze']
        raise ValueError(f"Unsupported gaze estimation method: '{method}'. Available: {available}")


__all__ = ['BaseGazeEstimator', 'GazeResult', 'MobileGaze', 'create_gaze_estimator']
