# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from typing_extensions import deprecated

from uniface.types import HeadPoseResult

from .base import BaseHeadPoseEstimator
from .models import HeadPose


@deprecated(
    'create_head_pose_estimator() is deprecated and will be removed in uniface 4.0. '
    'Instantiate the estimator class directly, e.g. '
    '`from uniface.headpose import HeadPose; HeadPose(**kwargs)`.'
)
def create_head_pose_estimator(method: str = 'headpose', **kwargs) -> BaseHeadPoseEstimator:
    """
    Factory function to create head pose estimators.

    .. deprecated:: 3.7.0
        Use ``HeadPose`` directly. This factory will be removed in uniface 4.0.

    This function initializes and returns a head pose estimator instance based on the
    specified method. It acts as a high-level interface to the underlying model classes.

    Args:
        method (str): The head pose estimation method to use.
            Options: 'headpose' (default).
        **kwargs: Model-specific parameters passed to the estimator's constructor.
            For example, `model_name` can be used to select a specific
            backbone from `HeadPoseWeights` enum (RESNET18, RESNET34, RESNET50,
            MOBILENET_V2, MOBILENET_V3_SMALL, MOBILENET_V3_LARGE).

    Returns:
        BaseHeadPoseEstimator: An initialized head pose estimator instance ready for use.

    Raises:
        ValueError: If the specified `method` is not supported.

    Examples:
        >>> # Create the default head pose estimator (ResNet18 backbone)
        >>> from uniface.headpose import HeadPose
        >>> estimator = HeadPose()

        >>> # Create with MobileNetV2 backbone
        >>> from uniface.constants import HeadPoseWeights
        >>> estimator = HeadPose(model_name=HeadPoseWeights.MOBILENET_V2)

        >>> # Use the estimator
        >>> result = estimator.estimate(face_crop)
        >>> print(f'Pitch: {result.pitch:.1f}°, Yaw: {result.yaw:.1f}°, Roll: {result.roll:.1f}°')
    """
    method = method.lower()

    if method in ('headpose', 'head_pose', '6drepnet'):
        return HeadPose(**kwargs)
    else:
        available = ['headpose']
        raise ValueError(f"Unsupported head pose estimation method: '{method}'. Available: {available}")


__all__ = ['BaseHeadPoseEstimator', 'HeadPose', 'HeadPoseResult', 'create_head_pose_estimator']
