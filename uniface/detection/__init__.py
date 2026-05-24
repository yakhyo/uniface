# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from typing import Any

from typing_extensions import deprecated

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


@deprecated(
    'create_detector() is deprecated and will be removed in uniface 4.0. '
    'Instantiate the detector class directly, e.g. '
    '`from uniface.detection import SCRFD; SCRFD(**kwargs)`.'
)
def create_detector(method: str = 'retinaface', **kwargs: Any) -> BaseDetector:
    """Factory function to create face detectors.

    .. deprecated:: 3.7.0
        Use the detector class directly (``RetinaFace``, ``SCRFD``,
        ``YOLOv5Face``, ``YOLOv8Face``). This factory will be removed in
        uniface 4.0.

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
        >>> from uniface.detection import RetinaFace
        >>> detector = RetinaFace()

        >>> # SCRFD detector with custom parameters
        >>> from uniface.detection import SCRFD
        >>> from uniface.constants import SCRFDWeights
        >>> detector = SCRFD(model_name=SCRFDWeights.SCRFD_10G_KPS, confidence_threshold=0.8, input_size=(640, 640))

        >>> # YOLOv8-Face detector
        >>> from uniface.detection import YOLOv8Face
        >>> from uniface.constants import YOLOv8FaceWeights
        >>> detector = YOLOv8Face(model_name=YOLOv8FaceWeights.YOLOV8N, confidence_threshold=0.5)
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


@deprecated(
    'list_available_detectors() is deprecated and will be removed in uniface 4.0. '
    'Import the detector classes directly from `uniface.detection` instead.'
)
def list_available_detectors() -> dict[str, dict[str, Any]]:
    """List all available detection methods with their descriptions and parameters.

    .. deprecated:: 3.7.0
        Import the detector classes (``RetinaFace``, ``SCRFD``, ``YOLOv5Face``,
        ``YOLOv8Face``) directly from ``uniface.detection``. This helper will be
        removed in uniface 4.0.

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
