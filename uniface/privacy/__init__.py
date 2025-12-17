# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from typing import Optional

import numpy as np

from .blur import BlurFace


def anonymize_faces(
    image: np.ndarray,
    detector: Optional[object] = None,
    method: str = 'pixelate',
    blur_strength: float = 3.0,
    pixel_blocks: int = 10,
    conf_thresh: float = 0.5,
    **kwargs,
) -> np.ndarray:
    """One-line face anonymization with automatic detection.

    Args:
        image (np.ndarray): Input image (BGR format).
        detector: Face detector instance. Creates RetinaFace if None.
        method (str): Blur method name. Defaults to 'pixelate'.
        blur_strength (float): Blur intensity. Defaults to 3.0.
        pixel_blocks (int): Block count for pixelate. Defaults to 10.
        conf_thresh (float): Detection confidence threshold. Defaults to 0.5.
        **kwargs: Additional detector arguments.

    Returns:
        np.ndarray: Anonymized image.

    Example:
        >>> from uniface.privacy import anonymize_faces
        >>> anonymized = anonymize_faces(image, method='pixelate')
    """
    if detector is None:
        try:
            from uniface import RetinaFace

            detector = RetinaFace(conf_thresh=conf_thresh, **kwargs)
        except ImportError as err:
            raise ImportError('Could not import RetinaFace. Please ensure UniFace is properly installed.') from err

    faces = detector.detect(image)
    blurrer = BlurFace(method=method, blur_strength=blur_strength, pixel_blocks=pixel_blocks)
    return blurrer.anonymize(image, faces)


__all__ = ['BlurFace', 'anonymize_faces']
