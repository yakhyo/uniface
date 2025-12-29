# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from uniface.face_utils import compute_similarity

__all__ = ['Face']


@dataclass(slots=True)
class Face:
    """
    Detected face with analysis results.

    This dataclass represents a single detected face along with optional
    analysis results such as embeddings, age, gender, and race predictions.

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

    # Required attributes
    bbox: np.ndarray
    confidence: float
    landmarks: np.ndarray

    # Optional attributes
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
