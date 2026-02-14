# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from typing import Any

from uniface.constants import (
    RetinaFaceWeights,
    SCRFDWeights,
    YOLOv5FaceWeights,
    YOLOv8FaceWeights,
)

from .base import BaseDetector
from .retinaface import RetinaFace
from .scrfd import SCRFD
from .yolov5 import YOLOv5Face
from .yolov8 import YOLOv8Face


def create_detector(method: str = 'retinaface', **kwargs: Any) -> BaseDetector:
    """Factory function to create face detectors.

    Args:
        method: Detection method. Options:
            - 'retinaface': RetinaFace detector (default)
            - 'scrfd': SCRFD detector (fast and accurate)
            - 'yolov5face': YOLOv5-Face detector (accurate with landmarks)
            - 'yolov8face': YOLOv8-Face detector (anchor-free, accurate)
        **kwargs: Detector-specific parameters.

    Returns:
        Initialized detector instance.

    Raises:
        ValueError: If method is not supported.

    Example:
        >>> # Basic usage
        >>> detector = create_detector('retinaface')

        >>> # SCRFD detector with custom parameters
        >>> from uniface.constants import SCRFDWeights
        >>> detector = create_detector(
        ...     'scrfd', model_name=SCRFDWeights.SCRFD_10G_KPS, confidence_threshold=0.8, input_size=(640, 640)
        ... )

        >>> # YOLOv8-Face detector
        >>> from uniface.constants import YOLOv8FaceWeights
        >>> detector = create_detector('yolov8face', model_name=YOLOv8FaceWeights.YOLOV8N, confidence_threshold=0.5)
    """
    method = method.lower()

    if method == 'retinaface':
        return RetinaFace(**kwargs)

    elif method == 'scrfd':
        return SCRFD(**kwargs)

    elif method == 'yolov5face':
        return YOLOv5Face(**kwargs)

    elif method == 'yolov8face':
        return YOLOv8Face(**kwargs)

    else:
        available_methods = ['retinaface', 'scrfd', 'yolov5face', 'yolov8face']
        raise ValueError(f"Unsupported detection method: '{method}'. Available methods: {available_methods}")


def list_available_detectors() -> dict[str, dict[str, Any]]:
    """List all available detection methods with their descriptions and parameters.

    Returns:
        Dictionary mapping detector names to their information including
        description, landmark support, paper reference, and default parameters.
    """
    return {
        'retinaface': {
            'description': 'RetinaFace detector with high accuracy',
            'supports_landmarks': True,
            'paper': 'https://arxiv.org/abs/1905.00641',
            'default_params': {
                'model_name': RetinaFaceWeights.MNET_V2.value,
                'confidence_threshold': 0.5,
                'nms_threshold': 0.4,
                'input_size': (640, 640),
            },
        },
        'scrfd': {
            'description': 'SCRFD detector - fast and accurate with efficient architecture',
            'supports_landmarks': True,
            'paper': 'https://arxiv.org/abs/2105.04714',
            'default_params': {
                'model_name': SCRFDWeights.SCRFD_10G_KPS.value,
                'confidence_threshold': 0.5,
                'nms_threshold': 0.4,
                'input_size': (640, 640),
            },
        },
        'yolov5face': {
            'description': 'YOLOv5-Face detector - accurate face detection with landmarks',
            'supports_landmarks': True,
            'paper': 'https://arxiv.org/abs/2105.12931',
            'default_params': {
                'model_name': YOLOv5FaceWeights.YOLOV5S.value,
                'confidence_threshold': 0.6,
                'nms_threshold': 0.5,
                'input_size': 640,
            },
        },
        'yolov8face': {
            'description': 'YOLOv8-Face detector - anchor-free design with DFL for accurate detection',
            'supports_landmarks': True,
            'paper': 'https://github.com/derronqi/yolov8-face',
            'default_params': {
                'model_name': YOLOv8FaceWeights.YOLOV8N.value,
                'confidence_threshold': 0.5,
                'nms_threshold': 0.45,
                'input_size': 640,
            },
        },
    }


__all__ = [
    'SCRFD',
    'BaseDetector',
    'RetinaFace',
    'YOLOv5Face',
    'YOLOv8Face',
    'create_detector',
    'list_available_detectors',
]
