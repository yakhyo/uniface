# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from uniface.face_utils import compute_similarity

__all__ = ['Face']


@dataclass
class Face:
    """Detected face with analysis results."""

    bbox: np.ndarray
    confidence: float
    landmarks: np.ndarray
    embedding: Optional[np.ndarray] = None
    age: Optional[int] = None
    gender_id: Optional[int] = None  # 0: Female, 1: Male

    def compute_similarity(self, other: 'Face') -> float:
        """Compute cosine similarity with another face."""
        if self.embedding is None or other.embedding is None:
            raise ValueError('Both faces must have embeddings for similarity computation')
        return float(compute_similarity(self.embedding, other.embedding))

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @property
    def gender(self) -> str:
        """Get gender as a string label (Female or Male)."""
        if self.gender_id is None:
            return None
        return 'Female' if self.gender_id == 0 else 'Male'

    def __repr__(self) -> str:
        parts = [f'Face(confidence={self.confidence:.3f}']
        if self.age is not None:
            parts.append(f'age={self.age}')
        if self.gender_id is not None:
            parts.append(f'gender={self.gender}')
        if self.embedding is not None:
            parts.append(f'embedding_dim={self.embedding.shape[0]}')
        return ', '.join(parts) + ')'
