# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from typing import List, Optional

import numpy as np

from uniface.attribute.age_gender import AgeGender
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
    ) -> None:
        self.detector = detector
        self.recognizer = recognizer
        self.age_gender = age_gender

        Logger.info(f'Initialized FaceAnalyzer with detector={detector.__class__.__name__}')
        if recognizer:
            Logger.info(f'  - Recognition enabled: {recognizer.__class__.__name__}')
        if age_gender:
            Logger.info(f'  - Age/Gender enabled: {age_gender.__class__.__name__}')

    def analyze(self, image: np.ndarray) -> List[Face]:
        """Analyze faces in an image."""
        detections = self.detector.detect(image)
        Logger.debug(f'Detected {len(detections)} face(s)')

        faces = []
        for idx, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            landmarks = detection['landmarks']

            embedding = None
            if self.recognizer is not None:
                try:
                    embedding = self.recognizer.get_normalized_embedding(image, landmarks)
                    Logger.debug(f'  Face {idx + 1}: Extracted embedding with shape {embedding.shape}')
                except Exception as e:
                    Logger.warning(f'  Face {idx + 1}: Failed to extract embedding: {e}')

            age, gender = None, None
            if self.age_gender is not None:
                try:
                    gender, age = self.age_gender.predict(image, bbox)
                    Logger.debug(f'  Face {idx + 1}: Age={age}, Gender={gender}')
                except Exception as e:
                    Logger.warning(f'  Face {idx + 1}: Failed to predict age/gender: {e}')

            face = Face(
                bbox=bbox,
                confidence=confidence,
                landmarks=landmarks,
                embedding=embedding,
                age=age,
                gender=gender,
            )
            faces.append(face)

        Logger.info(f'Analysis complete: {len(faces)} face(s) processed')
        return faces

    def __repr__(self) -> str:
        parts = [f'FaceAnalyzer(detector={self.detector.__class__.__name__}']
        if self.recognizer:
            parts.append(f'recognizer={self.recognizer.__class__.__name__}')
        if self.age_gender:
            parts.append(f'age_gender={self.age_gender.__class__.__name__}')
        return ', '.join(parts) + ')'
