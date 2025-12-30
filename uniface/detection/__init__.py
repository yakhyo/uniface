# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from typing import Any

import numpy as np

from uniface.types import Face

from .base import BaseDetector
from .retinaface import RetinaFace
from .scrfd import SCRFD
from .yolov5 import YOLOv5Face

# Global cache for detector instances (keyed by method name + config hash)
_detector_cache: dict[str, BaseDetector] = {}


def detect_faces(image: np.ndarray, method: str = 'retinaface', **kwargs: Any) -> list[Face]:
    """High-level face detection function.

    Detects faces in an image using the specified detection method.
    Results are cached for repeated calls with the same configuration.

    Args:
        image: Input image as numpy array with shape (H, W, C) in BGR format.
        method: Detection method to use. Options: 'retinaface', 'scrfd', 'yolov5face'.
        **kwargs: Additional arguments passed to the detector.

    Returns:
        A list of Face objects, each containing:
            - bbox: [x1, y1, x2, y2] bounding box coordinates.
            - confidence: The confidence score of the detection.
            - landmarks: 5-point facial landmarks with shape (5, 2).

    Example:
        >>> from uniface import detect_faces
        >>> import cv2
        >>> image = cv2.imread('your_image.jpg')
        >>> faces = detect_faces(image, method='retinaface', confidence_threshold=0.8)
        >>> for face in faces:
        ...     print(f'Found face with confidence: {face.confidence}')
        ...     print(f'BBox: {face.bbox}')
    """
    method_name = method.lower()

    sorted_kwargs = sorted(kwargs.items())
    cache_key = f'{method_name}_{sorted_kwargs!s}'

    if cache_key not in _detector_cache:
        # Pass kwargs to create the correctly configured detector
        _detector_cache[cache_key] = create_detector(method, **kwargs)

    detector = _detector_cache[cache_key]
    return detector.detect(image)


def create_detector(method: str = 'retinaface', **kwargs: Any) -> BaseDetector:
    """Factory function to create face detectors.

    Args:
        method: Detection method. Options:
            - 'retinaface': RetinaFace detector (default)
            - 'scrfd': SCRFD detector (fast and accurate)
            - 'yolov5face': YOLOv5-Face detector (accurate with landmarks)
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

        >>> # RetinaFace detector
        >>> from uniface.constants import RetinaFaceWeights
        >>> detector = create_detector(
        ...     'retinaface', model_name=RetinaFaceWeights.MNET_V2, confidence_threshold=0.8, nms_threshold=0.4
        ... )
    """
    method = method.lower()

    if method == 'retinaface':
        return RetinaFace(**kwargs)

    elif method == 'scrfd':
        return SCRFD(**kwargs)

    elif method == 'yolov5face':
        return YOLOv5Face(**kwargs)

    else:
        available_methods = ['retinaface', 'scrfd', 'yolov5face']
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
                'model_name': 'mnet_v2',
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
                'model_name': 'scrfd_10g_kps',
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
                'model_name': 'yolov5s_face',
                'confidence_threshold': 0.25,
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
    'create_detector',
    'detect_faces',
    'list_available_detectors',
]
