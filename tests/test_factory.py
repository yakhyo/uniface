# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Tests for factory functions (create_detector, create_recognizer, etc.)."""

from __future__ import annotations

import numpy as np
import pytest

from uniface import (
    create_detector,
    create_landmarker,
    create_recognizer,
    detect_faces,
    list_available_detectors,
)
from uniface.constants import RetinaFaceWeights, SCRFDWeights


# create_detector tests
def test_create_detector_retinaface():
    """
    Test creating a RetinaFace detector using factory function.
    """
    detector = create_detector('retinaface')
    assert detector is not None, 'Failed to create RetinaFace detector'


def test_create_detector_scrfd():
    """
    Test creating a SCRFD detector using factory function.
    """
    detector = create_detector('scrfd')
    assert detector is not None, 'Failed to create SCRFD detector'


def test_create_detector_with_config():
    """
    Test creating detector with custom configuration.
    """
    detector = create_detector(
        'retinaface',
        model_name=RetinaFaceWeights.MNET_V2,
        confidence_threshold=0.8,
        nms_threshold=0.3,
    )
    assert detector is not None, 'Failed to create detector with custom config'


def test_create_detector_invalid_method():
    """
    Test that invalid detector method raises an error.
    """
    with pytest.raises((ValueError, KeyError)):
        create_detector('invalid_method')


def test_create_detector_scrfd_with_model():
    """
    Test creating SCRFD detector with specific model.
    """
    detector = create_detector('scrfd', model_name=SCRFDWeights.SCRFD_10G_KPS, confidence_threshold=0.5)
    assert detector is not None, 'Failed to create SCRFD with specific model'


# create_recognizer tests
def test_create_recognizer_arcface():
    """
    Test creating an ArcFace recognizer using factory function.
    """
    recognizer = create_recognizer('arcface')
    assert recognizer is not None, 'Failed to create ArcFace recognizer'


def test_create_recognizer_mobileface():
    """
    Test creating a MobileFace recognizer using factory function.
    """
    recognizer = create_recognizer('mobileface')
    assert recognizer is not None, 'Failed to create MobileFace recognizer'


def test_create_recognizer_sphereface():
    """
    Test creating a SphereFace recognizer using factory function.
    """
    recognizer = create_recognizer('sphereface')
    assert recognizer is not None, 'Failed to create SphereFace recognizer'


def test_create_recognizer_invalid_method():
    """
    Test that invalid recognizer method raises an error.
    """
    with pytest.raises((ValueError, KeyError)):
        create_recognizer('invalid_method')


# create_landmarker tests
def test_create_landmarker():
    """
    Test creating a Landmark106 detector using factory function.
    """
    landmarker = create_landmarker('2d106det')
    assert landmarker is not None, 'Failed to create Landmark106 detector'


def test_create_landmarker_default():
    """
    Test creating landmarker with default parameters.
    """
    landmarker = create_landmarker()
    assert landmarker is not None, 'Failed to create default landmarker'


def test_create_landmarker_invalid_method():
    """
    Test that invalid landmarker method raises an error.
    """
    with pytest.raises((ValueError, KeyError)):
        create_landmarker('invalid_method')


# detect_faces tests
def test_detect_faces_retinaface():
    """
    Test high-level detect_faces function with RetinaFace.
    """
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    faces = detect_faces(mock_image, method='retinaface')

    assert isinstance(faces, list), 'detect_faces should return a list'


def test_detect_faces_scrfd():
    """
    Test high-level detect_faces function with SCRFD.
    """
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    faces = detect_faces(mock_image, method='scrfd')

    assert isinstance(faces, list), 'detect_faces should return a list'


def test_detect_faces_with_threshold():
    """
    Test detect_faces with custom confidence threshold.
    """
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    faces = detect_faces(mock_image, method='retinaface', confidence_threshold=0.8)

    assert isinstance(faces, list), 'detect_faces should return a list'

    # All detections should respect threshold
    for face in faces:
        assert face.confidence >= 0.8, 'All detections should meet confidence threshold'


def test_detect_faces_default_method():
    """
    Test detect_faces with default method (should use retinaface).
    """
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    faces = detect_faces(mock_image)  # No method specified

    assert isinstance(faces, list), 'detect_faces should return a list with default method'


def test_detect_faces_empty_image():
    """
    Test detect_faces on a blank image.
    """
    empty_image = np.zeros((640, 640, 3), dtype=np.uint8)
    faces = detect_faces(empty_image, method='retinaface')

    assert isinstance(faces, list), 'Should return a list even for empty image'
    assert len(faces) == 0, 'Should detect no faces in blank image'


# list_available_detectors tests
def test_list_available_detectors():
    """
    Test that list_available_detectors returns a dictionary.
    """
    detectors = list_available_detectors()

    assert isinstance(detectors, dict), 'Should return a dictionary of detectors'
    assert len(detectors) > 0, 'Should have at least one detector available'


def test_list_available_detectors_contents():
    """
    Test that list includes known detectors.
    """
    detectors = list_available_detectors()

    # Should include at least these detectors
    assert 'retinaface' in detectors, "Should include 'retinaface'"
    assert 'scrfd' in detectors, "Should include 'scrfd'"


# Integration tests
def test_detector_inference_from_factory():
    """
    Test that detector created from factory can perform inference.
    """
    detector = create_detector('retinaface')
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    faces = detector.detect(mock_image)
    assert isinstance(faces, list), 'Detector should return list of faces'


def test_recognizer_inference_from_factory():
    """
    Test that recognizer created from factory can perform inference.
    """
    recognizer = create_recognizer('arcface')
    mock_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

    embedding = recognizer.get_embedding(mock_image)
    assert embedding is not None, 'Recognizer should return embedding'
    assert embedding.shape[1] == 512, 'Should return 512-dimensional embedding'


def test_landmarker_inference_from_factory():
    """
    Test that landmarker created from factory can perform inference.
    """
    landmarker = create_landmarker('2d106det')
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    mock_bbox = [100, 100, 300, 300]

    landmarks = landmarker.get_landmarks(mock_image, mock_bbox)
    assert landmarks is not None, 'Landmarker should return landmarks'
    assert landmarks.shape == (106, 2), 'Should return 106 landmarks'


def test_multiple_detector_creation():
    """
    Test that multiple detectors can be created independently.
    """
    detector1 = create_detector('retinaface')
    detector2 = create_detector('scrfd')

    assert detector1 is not None
    assert detector2 is not None
    assert detector1 is not detector2, 'Should create separate instances'


def test_detector_with_different_configs():
    """
    Test creating multiple detectors with different configurations.
    """
    detector_high_thresh = create_detector('retinaface', confidence_threshold=0.9)
    detector_low_thresh = create_detector('retinaface', confidence_threshold=0.3)

    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    faces_high = detector_high_thresh.detect(mock_image)
    faces_low = detector_low_thresh.detect(mock_image)

    # Both should work
    assert isinstance(faces_high, list)
    assert isinstance(faces_low, list)


def test_factory_returns_correct_types():
    """
    Test that factory functions return instances of the correct types.
    """
    from uniface import ArcFace, Landmark106, RetinaFace

    detector = create_detector('retinaface')
    recognizer = create_recognizer('arcface')
    landmarker = create_landmarker('2d106det')

    assert isinstance(detector, RetinaFace), 'Should return RetinaFace instance'
    assert isinstance(recognizer, ArcFace), 'Should return ArcFace instance'
    assert isinstance(landmarker, Landmark106), 'Should return Landmark106 instance'
