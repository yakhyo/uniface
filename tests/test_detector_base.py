# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import numpy as np
import pytest

from uniface.analyzer import FaceAnalyzer
from uniface.detection import SCRFD, BlazeFace, CenterFace, RetinaFace, YOLOv5Face, YOLOv8Face
from uniface.detection.base import BaseDetector
from uniface.types import Face

LANDMARK_DETECTORS = [SCRFD, RetinaFace, CenterFace, YOLOv5Face, YOLOv8Face, BlazeFace]
ALIGNMENT_DETECTORS = [SCRFD, RetinaFace, CenterFace, YOLOv5Face, YOLOv8Face]


class BoxOnlyDetector(BaseDetector):
    """A subclass that declares nothing, like a third-party boxes-only detector."""

    def detect(self, image: np.ndarray, **kwargs) -> list[Face]:
        return []

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return image

    def postprocess(self, outputs, **kwargs):
        return outputs


def test_capability_flags_are_opt_in():
    """A subclass that declares nothing must not claim capabilities it does not have."""
    detector = BoxOnlyDetector()

    assert BaseDetector.supports_landmarks is False
    assert BaseDetector.supports_alignment is False
    assert detector.supports_landmarks is False
    assert detector.supports_alignment is False
    assert detector.get_info()['supports_landmarks'] is False
    assert detector.get_info()['supports_alignment'] is False


def test_analyzer_disables_recognition_for_box_only_detector():
    """A boxes-only detector must take the recognition-disabled branch, not load ArcFace."""
    analyzer = FaceAnalyzer(detector=BoxOnlyDetector())

    assert analyzer.recognizer is None


@pytest.mark.parametrize('detector_cls', LANDMARK_DETECTORS, ids=lambda c: c.__name__)
def test_builtin_detectors_declare_landmarks(detector_cls):
    assert detector_cls.supports_landmarks is True


@pytest.mark.parametrize('detector_cls', ALIGNMENT_DETECTORS, ids=lambda c: c.__name__)
def test_builtin_detectors_declare_alignment(detector_cls):
    assert detector_cls.supports_alignment is True


def test_blazeface_declares_landmarks_without_alignment():
    assert BlazeFace.supports_landmarks is True
    assert BlazeFace.supports_alignment is False
