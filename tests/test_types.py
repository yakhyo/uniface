# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Tests for UniFace type definitions (dataclasses)."""

from __future__ import annotations

import numpy as np
import pytest

from uniface.types import AttributeResult, EmotionResult, Face, GazeResult, SpoofingResult


class TestGazeResult:
    """Tests for GazeResult dataclass."""

    def test_creation(self):
        result = GazeResult(pitch=0.1, yaw=-0.2)
        assert result.pitch == 0.1
        assert result.yaw == -0.2

    def test_immutability(self):
        result = GazeResult(pitch=0.1, yaw=-0.2)
        with pytest.raises(AttributeError):
            result.pitch = 0.5  # type: ignore

    def test_repr(self):
        result = GazeResult(pitch=0.1234, yaw=-0.5678)
        repr_str = repr(result)
        assert 'GazeResult' in repr_str
        assert '0.1234' in repr_str
        assert '-0.5678' in repr_str

    def test_equality(self):
        result1 = GazeResult(pitch=0.1, yaw=-0.2)
        result2 = GazeResult(pitch=0.1, yaw=-0.2)
        assert result1 == result2

    def test_hashable(self):
        """Frozen dataclasses should be hashable."""
        result = GazeResult(pitch=0.1, yaw=-0.2)
        # Should not raise
        hash(result)
        # Can be used in sets/dicts
        result_set = {result}
        assert result in result_set


class TestSpoofingResult:
    """Tests for SpoofingResult dataclass."""

    def test_creation_real(self):
        result = SpoofingResult(is_real=True, confidence=0.95)
        assert result.is_real is True
        assert result.confidence == 0.95

    def test_creation_fake(self):
        result = SpoofingResult(is_real=False, confidence=0.87)
        assert result.is_real is False
        assert result.confidence == 0.87

    def test_immutability(self):
        result = SpoofingResult(is_real=True, confidence=0.95)
        with pytest.raises(AttributeError):
            result.is_real = False  # type: ignore

    def test_repr_real(self):
        result = SpoofingResult(is_real=True, confidence=0.9512)
        repr_str = repr(result)
        assert 'SpoofingResult' in repr_str
        assert 'Real' in repr_str
        assert '0.9512' in repr_str

    def test_repr_fake(self):
        result = SpoofingResult(is_real=False, confidence=0.8765)
        repr_str = repr(result)
        assert 'Fake' in repr_str

    def test_hashable(self):
        result = SpoofingResult(is_real=True, confidence=0.95)
        hash(result)


class TestEmotionResult:
    """Tests for EmotionResult dataclass."""

    def test_creation(self):
        result = EmotionResult(emotion='Happy', confidence=0.92)
        assert result.emotion == 'Happy'
        assert result.confidence == 0.92

    def test_immutability(self):
        result = EmotionResult(emotion='Sad', confidence=0.75)
        with pytest.raises(AttributeError):
            result.emotion = 'Happy'  # type: ignore

    def test_repr(self):
        result = EmotionResult(emotion='Angry', confidence=0.8123)
        repr_str = repr(result)
        assert 'EmotionResult' in repr_str
        assert 'Angry' in repr_str
        assert '0.8123' in repr_str

    def test_various_emotions(self):
        emotions = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry']
        for emotion in emotions:
            result = EmotionResult(emotion=emotion, confidence=0.5)
            assert result.emotion == emotion

    def test_hashable(self):
        result = EmotionResult(emotion='Happy', confidence=0.92)
        hash(result)


class TestAttributeResult:
    """Tests for AttributeResult dataclass."""

    def test_age_gender_result(self):
        result = AttributeResult(gender=1, age=25)
        assert result.gender == 1
        assert result.age == 25
        assert result.age_group is None
        assert result.race is None
        assert result.sex == 'Male'

    def test_fairface_result(self):
        result = AttributeResult(gender=0, age_group='20-29', race='East Asian')
        assert result.gender == 0
        assert result.age is None
        assert result.age_group == '20-29'
        assert result.race == 'East Asian'
        assert result.sex == 'Female'

    def test_sex_property_female(self):
        result = AttributeResult(gender=0)
        assert result.sex == 'Female'

    def test_sex_property_male(self):
        result = AttributeResult(gender=1)
        assert result.sex == 'Male'

    def test_immutability(self):
        result = AttributeResult(gender=1, age=30)
        with pytest.raises(AttributeError):
            result.age = 31  # type: ignore

    def test_repr_age_gender(self):
        result = AttributeResult(gender=1, age=25)
        repr_str = repr(result)
        assert 'AttributeResult' in repr_str
        assert 'Male' in repr_str
        assert 'age=25' in repr_str

    def test_repr_fairface(self):
        result = AttributeResult(gender=0, age_group='30-39', race='White')
        repr_str = repr(result)
        assert 'Female' in repr_str
        assert 'age_group=30-39' in repr_str
        assert 'race=White' in repr_str

    def test_hashable(self):
        result = AttributeResult(gender=1, age=25)
        hash(result)


class TestFace:
    """Tests for Face dataclass."""

    @pytest.fixture
    def sample_face(self):
        return Face(
            bbox=np.array([100, 100, 200, 200]),
            confidence=0.95,
            landmarks=np.array([[120, 130], [180, 130], [150, 160], [130, 180], [170, 180]]),
        )

    def test_creation(self, sample_face):
        assert sample_face.confidence == 0.95
        assert sample_face.bbox.shape == (4,)
        assert sample_face.landmarks.shape == (5, 2)

    def test_optional_attributes_default_none(self, sample_face):
        assert sample_face.embedding is None
        assert sample_face.gender is None
        assert sample_face.age is None
        assert sample_face.age_group is None
        assert sample_face.race is None
        assert sample_face.emotion is None
        assert sample_face.emotion_confidence is None

    def test_mutability(self, sample_face):
        """Face should be mutable for FaceAnalyzer enrichment."""
        sample_face.gender = 1
        sample_face.age = 25
        sample_face.embedding = np.random.randn(512)

        assert sample_face.gender == 1
        assert sample_face.age == 25
        assert sample_face.embedding.shape == (512,)

    def test_sex_property_none(self, sample_face):
        assert sample_face.sex is None

    def test_sex_property_female(self, sample_face):
        sample_face.gender = 0
        assert sample_face.sex == 'Female'

    def test_sex_property_male(self, sample_face):
        sample_face.gender = 1
        assert sample_face.sex == 'Male'

    def test_bbox_xyxy(self, sample_face):
        bbox_xyxy = sample_face.bbox_xyxy
        np.testing.assert_array_equal(bbox_xyxy, [100, 100, 200, 200])

    def test_bbox_xywh(self, sample_face):
        bbox_xywh = sample_face.bbox_xywh
        np.testing.assert_array_equal(bbox_xywh, [100, 100, 100, 100])

    def test_to_dict(self, sample_face):
        result = sample_face.to_dict()
        assert isinstance(result, dict)
        assert 'bbox' in result
        assert 'confidence' in result
        assert 'landmarks' in result

    def test_repr_minimal(self, sample_face):
        repr_str = repr(sample_face)
        assert 'Face' in repr_str
        assert 'confidence=0.950' in repr_str

    def test_repr_with_attributes(self, sample_face):
        sample_face.gender = 1
        sample_face.age = 30
        sample_face.emotion = 'Happy'

        repr_str = repr(sample_face)
        assert 'age=30' in repr_str
        assert 'sex=Male' in repr_str
        assert 'emotion=Happy' in repr_str

    def test_compute_similarity_no_embeddings(self, sample_face):
        other_face = Face(
            bbox=np.array([50, 50, 150, 150]),
            confidence=0.90,
            landmarks=np.random.randn(5, 2),
        )
        with pytest.raises(ValueError, match='Both faces must have embeddings'):
            sample_face.compute_similarity(other_face)

    def test_compute_similarity_with_embeddings(self, sample_face):
        # Create normalized embeddings
        sample_face.embedding = np.random.randn(512)
        sample_face.embedding /= np.linalg.norm(sample_face.embedding)

        other_face = Face(
            bbox=np.array([50, 50, 150, 150]),
            confidence=0.90,
            landmarks=np.random.randn(5, 2),
        )
        other_face.embedding = np.random.randn(512)
        other_face.embedding /= np.linalg.norm(other_face.embedding)

        similarity = sample_face.compute_similarity(other_face)
        assert isinstance(similarity, float)
        assert -1 <= similarity <= 1

    def test_compute_similarity_same_embedding(self, sample_face):
        embedding = np.random.randn(512)
        embedding /= np.linalg.norm(embedding)
        sample_face.embedding = embedding.copy()

        other_face = Face(
            bbox=np.array([50, 50, 150, 150]),
            confidence=0.90,
            landmarks=np.random.randn(5, 2),
            embedding=embedding.copy(),
        )

        similarity = sample_face.compute_similarity(other_face)
        assert similarity == pytest.approx(1.0, abs=1e-5)
