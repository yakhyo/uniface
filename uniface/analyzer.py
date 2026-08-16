# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from typing import Any

import numpy as np

from uniface.attribute.base import BaseAttribute
from uniface.detection.base import BaseDetector
from uniface.log import Logger
from uniface.recognition.base import BaseRecognizer
from uniface.types import Face

__all__ = ['FaceAnalyzer']

_UNSET: Any = object()


class FaceAnalyzer:
    """Unified face analyzer combining detection, recognition, and per-face predictors.

    This class provides a high-level interface for face analysis by combining
    multiple components: face detection, recognition (embedding extraction),
    and an extensible list of per-face predictors (age, gender, race,
    emotion, etc.).

    Any `BaseAttribute` subclass can be passed
    via the `predictors` list.  Each predictor's `predict(image, face)`
    is called once per detected face, enriching the `Face` in-place.

    When called with no arguments, uses SCRFD (500M) for detection and
    ArcFace (MobileNet) for recognition — the smallest and fastest variants.

    Args:
        detector: Face detector instance. Defaults to `SCRFD(SCRFD_500M_KPS)`.
        recognizer: Face recognizer for extracting embeddings.
            Defaults to `ArcFace(MNET)`. Pass `None` to disable recognition.
        predictors: Optional list of `BaseAttribute` predictors to run on
            each detected face (e.g. `[AgeGender()]`).

    Example:
        >>> from uniface import FaceAnalyzer
        >>> analyzer = FaceAnalyzer()
        >>> faces = analyzer.analyze(image)

        >>> from uniface import FaceAnalyzer, AgeGender
        >>> analyzer = FaceAnalyzer(predictors=[AgeGender()])
        >>> faces = analyzer.analyze(image)
    """

    def __init__(
        self,
        *,
        detector: BaseDetector | None = None,
        recognizer: BaseRecognizer | None = _UNSET,
        predictors: list[BaseAttribute] | None = None,
    ) -> None:
        if detector is None:
            from uniface.constants import SCRFDWeights
            from uniface.detection import SCRFD

            detector = SCRFD(model_name=SCRFDWeights.SCRFD_500M_KPS)

        # Checked before the _UNSET branch below so an unusable recognizer is never loaded.
        if not getattr(detector, 'supports_alignment', False):
            if recognizer is _UNSET or recognizer is not None:
                Logger.warning(
                    f'{detector.__class__.__name__} does not produce alignment landmarks; '
                    'recognition disabled. Use SCRFD, RetinaFace, CenterFace, YOLOv5Face or '
                    'YOLOv8Face if you need embeddings.'
                )
            recognizer = None

        if recognizer is _UNSET:
            from uniface.recognition import ArcFace

            recognizer = ArcFace()

        self.detector = detector
        self.recognizer = recognizer
        self.predictors: list[BaseAttribute] = predictors or []

        Logger.info(f'Initialized FaceAnalyzer with detector={detector.__class__.__name__}')
        if recognizer:
            Logger.info(f'Recognition enabled: {recognizer.__class__.__name__}')
        for attr in self.predictors:
            Logger.info(f'Predictor enabled: {attr.__class__.__name__}')

    def analyze(self, image: np.ndarray) -> list[Face]:
        """Analyze faces in an image.

        Performs face detection, optionally extracts embeddings, and runs
        every registered predictor on each detected face.

        Args:
            image: Input image as numpy array with shape (H, W, C) in BGR format.

        Returns:
            List of Face objects with detection results and any predictor results.
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

            for attr in self.predictors:
                attr_name = attr.__class__.__name__
                try:
                    attr.predict(image, face)
                except Exception as e:
                    Logger.warning(f'Face {idx + 1}: {attr_name} prediction failed: {e}')

        return faces

    def __repr__(self) -> str:
        parts = [f'detector={self.detector.__class__.__name__}']
        if self.recognizer is not None:
            parts.append(f'recognizer={self.recognizer.__class__.__name__}')
        if self.predictors:
            attr_names = ', '.join(attr.__class__.__name__ for attr in self.predictors)
            parts.append(f'predictors=[{attr_names}]')
        return f'FaceAnalyzer({", ".join(parts)})'
