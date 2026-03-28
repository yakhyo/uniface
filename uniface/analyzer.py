# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from typing import Any

import numpy as np

from uniface.attribute.base import Attribute
from uniface.detection.base import BaseDetector
from uniface.log import Logger
from uniface.recognition.base import BaseRecognizer
from uniface.types import Face

__all__ = ['FaceAnalyzer']

_UNSET: Any = object()


class FaceAnalyzer:
    """Unified face analyzer combining detection, recognition, and attributes.

    This class provides a high-level interface for face analysis by combining
    multiple components: face detection, recognition (embedding extraction),
    and an extensible list of attribute predictors (age, gender, race,
    emotion, etc.).

    Any :class:`~uniface.attribute.base.Attribute` subclass can be passed
    via the ``attributes`` list.  Each predictor's ``predict(image, face)``
    is called once per detected face, enriching the :class:`Face` in-place.

    When called with no arguments, uses SCRFD (500M) for detection and
    ArcFace (MobileNet) for recognition — the smallest and fastest variants.

    Args:
        detector: Face detector instance. Defaults to ``SCRFD(SCRFD_500M_KPS)``.
        recognizer: Face recognizer for extracting embeddings.
            Defaults to ``ArcFace(MNET)``. Pass ``None`` to disable recognition.
        attributes: Optional list of ``Attribute`` predictors to run on
            each detected face (e.g. ``[AgeGender()]``).

    Examples:
        >>> from uniface import FaceAnalyzer
        >>> analyzer = FaceAnalyzer()
        >>> faces = analyzer.analyze(image)

        >>> from uniface import FaceAnalyzer, AgeGender
        >>> analyzer = FaceAnalyzer(attributes=[AgeGender()])
        >>> faces = analyzer.analyze(image)
    """

    def __init__(
        self,
        detector: BaseDetector | None = None,
        recognizer: BaseRecognizer | None = _UNSET,
        attributes: list[Attribute] | None = None,
    ) -> None:
        if detector is None:
            from uniface.constants import SCRFDWeights
            from uniface.detection import SCRFD

            detector = SCRFD(model_name=SCRFDWeights.SCRFD_500M_KPS)

        if recognizer is _UNSET:
            from uniface.recognition import ArcFace

            recognizer = ArcFace()

        self.detector = detector
        self.recognizer = recognizer
        self.attributes: list[Attribute] = attributes or []

        Logger.info(f'Initialized FaceAnalyzer with detector={detector.__class__.__name__}')
        if recognizer:
            Logger.info(f'Recognition enabled: {recognizer.__class__.__name__}')
        for attr in self.attributes:
            Logger.info(f'Attribute enabled: {attr.__class__.__name__}')

    def analyze(self, image: np.ndarray) -> list[Face]:
        """Analyze faces in an image.

        Performs face detection, optionally extracts embeddings, and runs
        every registered attribute predictor on each detected face.

        Args:
            image: Input image as numpy array with shape (H, W, C) in BGR format.

        Returns:
            List of Face objects with detection results and any predicted attributes.
        """
        faces = self.detector.detect(image)
        Logger.debug(f'Detected {len(faces)} face(s)')

        for idx, face in enumerate(faces):
            if self.recognizer is not None:
                try:
                    face.embedding = self.recognizer.get_normalized_embedding(image, face.landmarks)
                    Logger.debug(f'Face {idx + 1}: Extracted embedding with shape {face.embedding.shape}')
                except Exception as e:
                    Logger.warning(f'Face {idx + 1}: Failed to extract embedding: {e}')

            for attr in self.attributes:
                attr_name = attr.__class__.__name__
                try:
                    attr.predict(image, face)
                    Logger.debug(f'Face {idx + 1}: {attr_name} prediction succeeded')
                except Exception as e:
                    Logger.warning(f'Face {idx + 1}: {attr_name} prediction failed: {e}')

        Logger.info(f'Analysis complete: {len(faces)} face(s) processed')
        return faces

    def __repr__(self) -> str:
        parts = [f'FaceAnalyzer(detector={self.detector.__class__.__name__}']
        if self.recognizer:
            parts.append(f'recognizer={self.recognizer.__class__.__name__}')
        for attr in self.attributes:
            parts.append(f'{attr.__class__.__name__}')
        return ', '.join(parts) + ')'
