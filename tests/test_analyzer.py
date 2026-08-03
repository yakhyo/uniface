# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import numpy as np
import pytest

from uniface.analyzer import FaceAnalyzer
from uniface.attribute import BaseAttribute
from uniface.detection.base import BaseDetector
from uniface.types import Face


class StubDetector(BaseDetector):
    """Returns a fixed number of faces without any model or download."""

    supports_landmarks = True
    supports_alignment = True

    def __init__(self, num_faces: int = 2) -> None:
        super().__init__()
        self.num_faces = num_faces

    def detect(self, image: np.ndarray, **kwargs) -> list[Face]:
        return [
            Face(
                bbox=np.array([10.0 * i, 10.0, 50.0 + 10.0 * i, 60.0]),
                confidence=0.9,
                landmarks=np.zeros((5, 2)),
            )
            for i in range(self.num_faces)
        ]

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return image

    def postprocess(self, outputs, **kwargs):
        return outputs


class AgeStampPredictor(BaseAttribute):
    """Enriches each face in-place and counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def _initialize_model(self) -> None: ...

    def preprocess(self, image, *args):
        return image

    def postprocess(self, prediction):
        return prediction

    def predict(self, image: np.ndarray, face: Face) -> None:
        self.calls += 1
        face.age = 30


class FailingPredictor(BaseAttribute):
    """Always raises, to exercise the analyzer's per-predictor error handling."""

    def _initialize_model(self) -> None: ...

    def preprocess(self, image, *args):
        return image

    def postprocess(self, prediction):
        return prediction

    def predict(self, image: np.ndarray, face: Face) -> None:
        raise RuntimeError('boom')


@pytest.fixture
def image():
    return np.zeros((240, 320, 3), dtype=np.uint8)


def test_predictors_run_once_per_face(image):
    predictor = AgeStampPredictor()
    analyzer = FaceAnalyzer(detector=StubDetector(num_faces=3), recognizer=None, predictors=[predictor])

    faces = analyzer.analyze(image)

    assert predictor.calls == 3
    assert all(face.age == 30 for face in faces)


def test_failing_predictor_does_not_break_the_pipeline(image):
    """One bad predictor must not abort analysis or starve the others."""
    stamper = AgeStampPredictor()
    analyzer = FaceAnalyzer(
        detector=StubDetector(num_faces=2),
        recognizer=None,
        predictors=[FailingPredictor(), stamper],
    )

    faces = analyzer.analyze(image)

    assert len(faces) == 2
    assert stamper.calls == 2


def test_predictors_default_to_empty(image):
    analyzer = FaceAnalyzer(detector=StubDetector(), recognizer=None)

    assert analyzer.predictors == []
    assert len(analyzer.analyze(image)) == 2


def test_repr_lists_predictors():
    analyzer = FaceAnalyzer(detector=StubDetector(), recognizer=None, predictors=[AgeStampPredictor()])

    assert 'predictors=[AgeStampPredictor]' in repr(analyzer)
    assert 'recognizer' not in repr(analyzer)


def test_constructor_is_keyword_only():
    with pytest.raises(TypeError):
        FaceAnalyzer(StubDetector())
