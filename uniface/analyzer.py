# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import numpy as np

from uniface.attribute.base import Attribute
from uniface.detection.base import BaseDetector
from uniface.log import Logger
from uniface.recognition.base import BaseRecognizer
from uniface.types import Face

__all__ = ['FaceAnalyzer']


class FaceAnalyzer:
    """Unified face analyzer combining detection, recognition, and attributes.

    This class provides a high-level interface for face analysis by combining
    multiple components: face detection, recognition (embedding extraction),
    and an extensible list of attribute predictors (age, gender, race,
    emotion, etc.).

    Any :class:`~uniface.attribute.base.Attribute` subclass can be passed
    via the ``attributes`` list.  Each predictor's ``predict(image, face)``
    is called once per detected face, enriching the :class:`Face` in-place.

    Args:
        detector: Face detector instance for detecting faces in images.
        recognizer: Optional face recognizer for extracting embeddings.
        attributes: Optional list of ``Attribute`` predictors to run on
            each detected face (e.g. ``[AgeGender(), FairFace(), Emotion()]``).

    Example:
        >>> from uniface import RetinaFace, ArcFace, AgeGender, FaceAnalyzer
        >>> detector = RetinaFace()
        >>> recognizer = ArcFace()
        >>> analyzer = FaceAnalyzer(detector, recognizer=recognizer, attributes=[AgeGender()])
        >>> faces = analyzer.analyze(image)
    """

    def __init__(
        self,
        detector: BaseDetector,
        recognizer: BaseRecognizer | None = None,
        attributes: list[Attribute] | None = None,
    ) -> None:
        self.detector = detector
        self.recognizer = recognizer
        self.attributes: list[Attribute] = attributes or []

        Logger.info(f'Initialized FaceAnalyzer with detector={detector.__class__.__name__}')
        if recognizer:
            Logger.info(f'  - Recognition enabled: {recognizer.__class__.__name__}')
        for attr in self.attributes:
            Logger.info(f'  - Attribute enabled: {attr.__class__.__name__}')

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
                    Logger.debug(f'  Face {idx + 1}: Extracted embedding with shape {face.embedding.shape}')
                except Exception as e:
                    Logger.warning(f'  Face {idx + 1}: Failed to extract embedding: {e}')

            for attr in self.attributes:
                attr_name = attr.__class__.__name__
                try:
                    attr.predict(image, face)
                    Logger.debug(f'  Face {idx + 1}: {attr_name} prediction succeeded')
                except Exception as e:
                    Logger.warning(f'  Face {idx + 1}: {attr_name} prediction failed: {e}')

        Logger.info(f'Analysis complete: {len(faces)} face(s) processed')
        return faces

    def __repr__(self) -> str:
        parts = [f'FaceAnalyzer(detector={self.detector.__class__.__name__}']
        if self.recognizer:
            parts.append(f'recognizer={self.recognizer.__class__.__name__}')
        for attr in self.attributes:
            parts.append(f'{attr.__class__.__name__}')
        return ', '.join(parts) + ')'
