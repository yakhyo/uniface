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
    """
    Detected face with analysis results.
    """

    # Required attributes
    bbox: np.ndarray
    confidence: float
    landmarks: np.ndarray

    # Optional attributes
    embedding: Optional[np.ndarray] = None
    age: Optional[int] = None
    gender: Optional[int] = None  # 0 or 1

    def compute_similarity(self, other: 'Face') -> float:
        """Compute cosine similarity with another face."""
        if self.embedding is None or other.embedding is None:
            raise ValueError('Both faces must have embeddings for similarity computation')
        return float(compute_similarity(self.embedding, other.embedding))

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @property
    def sex(self) -> str:
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
        if self.gender is not None:
            parts.append(f'sex={self.sex}')
        if self.embedding is not None:
            parts.append(f'embedding_dim={self.embedding.shape[0]}')
        return ', '.join(parts) + ')'
