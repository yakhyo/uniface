# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from typing import List, Optional

import numpy as np

from uniface.attribute.age_gender import AgeGender
from uniface.attribute.fairface import FairFace
from uniface.detection.base import BaseDetector
from uniface.face import Face
from uniface.log import Logger
from uniface.recognition.base import BaseRecognizer

__all__ = ['FaceAnalyzer']


class FaceAnalyzer:
    """Unified face analyzer combining detection, recognition, and attributes."""

    def __init__(
        self,
        detector: BaseDetector,
        recognizer: Optional[BaseRecognizer] = None,
        age_gender: Optional[AgeGender] = None,
        fairface: Optional[FairFace] = None,
    ) -> None:
        self.detector = detector
        self.recognizer = recognizer
        self.age_gender = age_gender
        self.fairface = fairface

        Logger.info(f'Initialized FaceAnalyzer with detector={detector.__class__.__name__}')
        if recognizer:
            Logger.info(f'  - Recognition enabled: {recognizer.__class__.__name__}')
        if age_gender:
            Logger.info(f'  - Age/Gender enabled: {age_gender.__class__.__name__}')
        if fairface:
            Logger.info(f'  - FairFace enabled: {fairface.__class__.__name__}')

    def analyze(self, image: np.ndarray) -> List[Face]:
        """Analyze faces in an image."""
        faces = self.detector.detect(image)
        Logger.debug(f'Detected {len(faces)} face(s)')

        for idx, face in enumerate(faces):
            if self.recognizer is not None:
                try:
                    face.embedding = self.recognizer.get_normalized_embedding(image, face.landmarks)
                    Logger.debug(f'  Face {idx + 1}: Extracted embedding with shape {face.embedding.shape}')
                except Exception as e:
                    Logger.warning(f'  Face {idx + 1}: Failed to extract embedding: {e}')

            if self.age_gender is not None:
                try:
                    result = self.age_gender.predict(image, face.bbox)
                    face.gender = result.gender
                    face.age = result.age
                    Logger.debug(f'  Face {idx + 1}: Age={face.age}, Gender={face.sex}')
                except Exception as e:
                    Logger.warning(f'  Face {idx + 1}: Failed to predict age/gender: {e}')

            if self.fairface is not None:
                try:
                    result = self.fairface.predict(image, face.bbox)
                    face.gender = result.gender
                    face.age_group = result.age_group
                    face.race = result.race
                    Logger.debug(f'  Face {idx + 1}: AgeGroup={face.age_group}, Gender={face.sex}, Race={face.race}')
                except Exception as e:
                    Logger.warning(f'  Face {idx + 1}: Failed to predict FairFace attributes: {e}')

        Logger.info(f'Analysis complete: {len(faces)} face(s) processed')
        return faces

    def __repr__(self) -> str:
        parts = [f'FaceAnalyzer(detector={self.detector.__class__.__name__}']
        if self.recognizer:
            parts.append(f'recognizer={self.recognizer.__class__.__name__}')
        if self.age_gender:
            parts.append(f'age_gender={self.age_gender.__class__.__name__}')
        if self.fairface:
            parts.append(f'fairface={self.fairface.__class__.__name__}')
        return ', '.join(parts) + ')'
