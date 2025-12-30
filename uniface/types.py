# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Unified type definitions for UniFace.

This module centralizes all result dataclasses used across the library,
providing consistent and immutable return types for model predictions.

Note on mutability:
- Result dataclasses (GazeResult, SpoofingResult, EmotionResult, AttributeResult)
  are frozen (immutable) since they represent computation outputs that shouldn't change.
- Face dataclass is mutable because FaceAnalyzer enriches it with additional
  attributes (embedding, age, gender, etc.) after initial detection.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from uniface.face_utils import compute_similarity

__all__ = [
    'AttributeResult',
    'EmotionResult',
    'Face',
    'GazeResult',
    'SpoofingResult',
]


@dataclass(slots=True, frozen=True)
class GazeResult:
    """Result of gaze estimation.

    Attributes:
        pitch: Vertical gaze angle in radians (positive = up, negative = down).
        yaw: Horizontal gaze angle in radians (positive = right, negative = left).
    """

    pitch: float
    yaw: float

    def __repr__(self) -> str:
        return f'GazeResult(pitch={self.pitch:.4f}, yaw={self.yaw:.4f})'


@dataclass(slots=True, frozen=True)
class SpoofingResult:
    """Result of face anti-spoofing detection.

    Attributes:
        is_real: True if the face is real/live, False if fake/spoof.
        confidence: Confidence score for the prediction (0.0 to 1.0).
    """

    is_real: bool
    confidence: float

    def __repr__(self) -> str:
        label = 'Real' if self.is_real else 'Fake'
        return f'SpoofingResult({label}, confidence={self.confidence:.4f})'


@dataclass(slots=True, frozen=True)
class EmotionResult:
    """Result of emotion recognition.

    Attributes:
        emotion: Predicted emotion label (e.g., 'Happy', 'Sad', 'Angry').
        confidence: Confidence score for the prediction (0.0 to 1.0).
    """

    emotion: str
    confidence: float

    def __repr__(self) -> str:
        return f"EmotionResult('{self.emotion}', confidence={self.confidence:.4f})"


@dataclass(slots=True, frozen=True)
class AttributeResult:
    """Unified result structure for face attribute prediction.

    This dataclass provides a consistent return type across different attribute
    prediction models (e.g., AgeGender, FairFace), enabling interoperability
    and unified handling of results.

    Attributes:
        gender: Predicted gender (0=Female, 1=Male).
        age: Exact age in years. Provided by AgeGender model, None for FairFace.
        age_group: Age range string like "20-29". Provided by FairFace, None for AgeGender.
        race: Race/ethnicity label. Provided by FairFace only.

    Properties:
        sex: Gender as a human-readable string ("Female" or "Male").

    Examples:
        >>> # AgeGender result
        >>> result = AttributeResult(gender=1, age=25)
        >>> result.sex
        'Male'

        >>> # FairFace result
        >>> result = AttributeResult(gender=0, age_group='20-29', race='East Asian')
        >>> result.sex
        'Female'
    """

    gender: int
    age: int | None = None
    age_group: str | None = None
    race: str | None = None

    @property
    def sex(self) -> str:
        """Get gender as a string label (Female or Male)."""
        return 'Female' if self.gender == 0 else 'Male'

    def __repr__(self) -> str:
        parts = [f'gender={self.sex}']
        if self.age is not None:
            parts.append(f'age={self.age}')
        if self.age_group is not None:
            parts.append(f'age_group={self.age_group}')
        if self.race is not None:
            parts.append(f'race={self.race}')
        return f'AttributeResult({", ".join(parts)})'


@dataclass(slots=True)
class Face:
    """Detected face with analysis results.

    This dataclass represents a single detected face along with optional
    analysis results such as embeddings, age, gender, and race predictions.

    Note: This dataclass is mutable (not frozen) because FaceAnalyzer enriches
    Face objects with additional attributes after initial detection.

    Attributes:
        bbox: Bounding box coordinates [x1, y1, x2, y2].
        confidence: Detection confidence score.
        landmarks: Facial landmark coordinates (typically 5 points).
        embedding: Face embedding vector for recognition (optional).
        gender: Predicted gender, 0=Female, 1=Male (optional).
        age: Predicted exact age in years (optional, from AgeGender model).
        age_group: Predicted age range like "20-29" (optional, from FairFace).
        race: Predicted race/ethnicity (optional, from FairFace).
        emotion: Predicted emotion label (optional, from Emotion model).
        emotion_confidence: Confidence score for emotion prediction (optional).

    Properties:
        sex: Gender as a human-readable string ("Female" or "Male").
        bbox_xyxy: Bounding box in (x1, y1, x2, y2) format.
        bbox_xywh: Bounding box in (x1, y1, width, height) format.
    """

    # Required attributes (from detection)
    bbox: np.ndarray
    confidence: float
    landmarks: np.ndarray

    # Optional attributes (enriched by analyzers)
    embedding: np.ndarray | None = None
    gender: int | None = None
    age: int | None = None
    age_group: str | None = None
    race: str | None = None
    emotion: str | None = None
    emotion_confidence: float | None = None

    def compute_similarity(self, other: Face) -> float:
        """Compute cosine similarity with another face."""
        if self.embedding is None or other.embedding is None:
            raise ValueError('Both faces must have embeddings for similarity computation')
        return float(compute_similarity(self.embedding, other.embedding))

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @property
    def sex(self) -> str | None:
        """Get gender as a string label (Female or Male)."""
        if self.gender is None:
            return None
        return 'Female' if self.gender == 0 else 'Male'

    @property
    def bbox_xyxy(self) -> np.ndarray:
        """Get bounding box coordinates in (x1, y1, x2, y2) format."""
        return self.bbox.copy()

    @property
    def bbox_xywh(self) -> np.ndarray:
        """Get bounding box coordinates in (x1, y1, w, h) format."""
        return np.array([self.bbox[0], self.bbox[1], self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]])

    def __repr__(self) -> str:
        parts = [f'Face(confidence={self.confidence:.3f}']
        if self.age is not None:
            parts.append(f'age={self.age}')
        if self.age_group is not None:
            parts.append(f'age_group={self.age_group}')
        if self.gender is not None:
            parts.append(f'sex={self.sex}')
        if self.race is not None:
            parts.append(f'race={self.race}')
        if self.emotion is not None:
            parts.append(f'emotion={self.emotion}')
        if self.embedding is not None:
            parts.append(f'embedding_dim={self.embedding.shape[0]}')
        return ', '.join(parts) + ')'
