# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


import numpy as np
from typing import Tuple, Dict, Any, List

from .scrfd import SCRFD
from .base import BaseDetector
from .retinaface import RetinaFace

# Global cache for detector instances
_detector_cache: Dict[str, BaseDetector] = {}


def detect_faces(image: np.ndarray, method: str = 'retinaface', **kwargs) -> List[Dict[str, Any]]:
    """
    High-level face detection function.

    Args:
        image (np.ndarray): Input image as numpy array.
        method (str): Detection method to use. Options: 'retinaface', 'scrfd'.
        **kwargs: Additional arguments passed to the detector.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents a detected face and contains:
            - 'bbox' (List[float]): [x1, y1, x2, y2] bounding box coordinates.
            - 'confidence' (float): The confidence score of the detection.
            - 'landmarks' (List[List[float]]): 5-point facial landmarks.

    Example:
        >>> from uniface import detect_faces
        >>> image = cv2.imread("your_image.jpg")
        >>> faces = detect_faces(image, method='retinaface', conf_thresh=0.8)
        >>> for face in faces:
        ...     print(f"Found face with confidence: {face['confidence']}")
        ...     print(f"BBox: {face['bbox']}")
    """
    method_name = method.lower()
    
    sorted_kwargs = sorted(kwargs.items())
    cache_key = f"{method_name}_{str(sorted_kwargs)}"

    if cache_key not in _detector_cache:
        # Pass kwargs to create the correctly configured detector
        _detector_cache[cache_key] = create_detector(method, **kwargs)

    detector = _detector_cache[cache_key]
    return detector.detect(image)


def create_detector(method: str = 'retinaface', **kwargs) -> BaseDetector:
    """
    Factory function to create face detectors.

    Args:
        method (str): Detection method. Options:
            - 'retinaface': RetinaFace detector (default)
            - 'scrfd': SCRFD detector (fast and accurate)
        **kwargs: Detector-specific parameters

    Returns:
        BaseDetector: Initialized detector instance

    Raises:
        ValueError: If method is not supported

    Examples:
        >>> # Basic usage
        >>> detector = create_detector('retinaface')

        >>> # SCRFD detector with custom parameters
        >>> detector = create_detector(
        ...     'scrfd',
        ...     model_name=SCRFDWeights.SCRFD_10G_KPS,
        ...     conf_thresh=0.8,
        ...     input_size=(640, 640)
        ... )

        >>> # RetinaFace detector
        >>> detector = create_detector(
        ...     'retinaface',
        ...     model_name=RetinaFaceWeights.MNET_V2,
        ...     conf_thresh=0.8,
        ...     nms_thresh=0.4
        ... )
    """
    method = method.lower()

    if method == 'retinaface':
        return RetinaFace(**kwargs)

    elif method == 'scrfd':
        return SCRFD(**kwargs)

    else:
        available_methods = ['retinaface', 'scrfd']
        raise ValueError(
            f"Unsupported detection method: '{method}'. "
            f"Available methods: {available_methods}"
        )


def list_available_detectors() -> Dict[str, Dict[str, Any]]:
    """
    List all available detection methods with their descriptions and parameters.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of detector information
    """
    return {
        'retinaface': {
            'description': 'RetinaFace detector with high accuracy',
            'supports_landmarks': True,
            'paper': 'https://arxiv.org/abs/1905.00641',
            'default_params': {
                'model_name': 'mnet_v2',
                'conf_thresh': 0.5,
                'nms_thresh': 0.4,
                'input_size': (640, 640)
            }
        },
        'scrfd': {
            'description': 'SCRFD detector - fast and accurate with efficient architecture',
            'supports_landmarks': True,
            'paper': 'https://arxiv.org/abs/2105.04714',
            'default_params': {
                'model_name': 'scrfd_10g_kps',
                'conf_thresh': 0.5,
                'nms_thresh': 0.4,
                'input_size': (640, 640)
            }
        }
    }


__all__ = [
    'detect_faces',
    'create_detector',
    'list_available_detectors',
    'SCRFD',
    'RetinaFace',
    'BaseDetector',
]
