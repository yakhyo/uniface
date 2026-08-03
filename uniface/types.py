# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Unified type definitions for UniFace.

This module centralizes all result dataclasses used across the library,
providing consistent and immutable return types for model predictions.

Note on mutability:
- Result dataclasses (GazeResult, SpoofingResult, EmotionResult, DemographyResult)
  are frozen (immutable) since they represent computation outputs that shouldn't change.
- Face dataclass is mutable because FaceAnalyzer enriches it with additional
  attributes (embedding, age, gender, etc.) after initial detection.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import json

import numpy as np

from uniface.face_utils import compute_similarity

__all__ = [
    'DemographyResult',
    'EmotionResult',
    'Face',
    'FaceMeshResult',
    'FaceStateResult',
    'GazeResult',
    'HeadPoseResult',
    'QualityResult',
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
class HeadPoseResult:
    """Result of head pose estimation.

    Attributes:
        pitch: Rotation around X-axis in degrees (positive = looking down).
        yaw: Rotation around Y-axis in degrees (positive = looking right).
        roll: Rotation around Z-axis in degrees (positive = tilting clockwise).
    """

    pitch: float
    yaw: float
    roll: float

    def __repr__(self) -> str:
        return f'HeadPoseResult(pitch={self.pitch:.1f}, yaw={self.yaw:.1f}, roll={self.roll:.1f})'


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
class QualityResult:
    """Result of face image quality assessment.

    Attributes:
        score: Quality score in [0, 1]. Higher = better quality.
    """

    score: float

    def __repr__(self) -> str:
        return f'QualityResult(score={self.score:.4f})'


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
class DemographyResult:
    """Unified result structure for demographic attribute prediction.

    This dataclass provides a consistent return type across demographic
    prediction models (AgeGender, FairFace), enabling interoperability
    and unified handling of results.

    Attributes:
        gender: Predicted gender (0=Female, 1=Male).
        age: Exact age in years. Provided by AgeGender model, None for FairFace.
        age_group: Age range string like "20-29". Provided by FairFace, None for AgeGender.
        race: Race/ethnicity label. Provided by FairFace only.
        sex: Read-only. Gender as a human-readable string ("Female" or "Male").

    Example:
        >>> # AgeGender result
        >>> result = DemographyResult(gender=1, age=25)
        >>> result.sex
        'Male'

        >>> # FairFace result
        >>> result = DemographyResult(gender=0, age_group='20-29', race='East Asian')
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
        return f'DemographyResult({", ".join(parts)})'


@dataclass(slots=True, frozen=True)
class FaceStateResult:
    """Result of face state prediction (FaceAttribNet).

    Five independent binary probabilities in [0, 1], in the model's output
    order. Each value comes from its own classifier head, so the values do not
    sum to 1 and several can be high at once (a face can wear both sunglasses
    and a mask). Threshold each attribute separately; never argmax.

    Attributes:
        left_eye_open: Probability the left eye is open.
        right_eye_open: Probability the right eye is open.
        eyeglasses: Probability eyeglasses are present.
        mask: Probability a face mask is present.
        sunglasses: Probability sunglasses are present.
    """

    left_eye_open: float
    right_eye_open: float
    eyeglasses: float
    mask: float
    sunglasses: float

    def as_dict(self) -> dict[str, float]:
        """Return the attributes as a name -> probability mapping."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def labels(self, threshold: float = 0.5) -> list[str]:
        """Return the names of attributes whose probability exceeds `threshold`."""
        return [name for name, value in self.as_dict().items() if value > threshold]

    def __repr__(self) -> str:
        body = ', '.join(f'{name}={value:.4f}' for name, value in self.as_dict().items())
        return f'FaceStateResult({body})'


@dataclass(slots=True, frozen=True, eq=False)
class FaceMeshResult:
    """Result of dense facial landmark prediction (Face Mesh).

    Note on `eq=False`: this dataclass holds a numpy array, and the generated
    `__eq__`/`__hash__` would raise (`ValueError` on `==`, `TypeError`
    on `hash`). Identity comparison is used instead; compare `landmarks`
    directly with `np.allclose` if you need value semantics.

    Attributes:
        landmarks: Dense landmarks with shape (468, 3) or (478, 3), float32, depending
            on which `FaceMeshWeights` model produced them. `x`/`y` are in full-image
            pixel coordinates; `z` is relative depth on the same pixel scale (smaller is
            closer to the camera). Google trains `z` on synthetic data and excludes it
            from their own accuracy evaluation, so treat it as weaker than `x`/`y`.

            With the 478-point model the first 468 are the mesh, in the same order the
            468-point model produces, and 468-477 are the irises.
        score: Face-presence score in [0, 1].

            This value saturates. The network emits a raw logit that is typically
            20-40 for any plausible face, so after the sigmoid it is ~1.0 almost
            always. Treat it as confirmation that the model ran, not as a
            discriminative confidence, and do not threshold on it.
    """

    landmarks: np.ndarray
    score: float

    @property
    def points_2d(self) -> np.ndarray:
        """Get the landmarks without the depth component, shape (N, 2)."""
        return self.landmarks[:, :2]

    def __repr__(self) -> str:
        return f'FaceMeshResult(landmarks={self.landmarks.shape}, score={self.score:.4f})'


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
        left_eye_open: Probability the left eye is open (optional, from FaceAttribNet).
        right_eye_open: Probability the right eye is open (optional, from FaceAttribNet).
        eyeglasses: Probability eyeglasses are present (optional, from FaceAttribNet).
        mask: Probability a face mask is present (optional, from FaceAttribNet).
        sunglasses: Probability sunglasses are present (optional, from FaceAttribNet).
        quality: Face image quality score in [0, 1] (optional, from eDifFIQA).
        track_id: Persistent track ID assigned by BYTETracker (optional).
        sex: Read-only. Gender as a human-readable string ("Female" or "Male").
        bbox_xyxy: Read-only. Bounding box in (x1, y1, x2, y2) format.
        bbox_xywh: Read-only. Bounding box in (x1, y1, width, height) format.
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
    left_eye_open: float | None = None
    right_eye_open: float | None = None
    eyeglasses: float | None = None
    mask: float | None = None
    sunglasses: float | None = None
    quality: float | None = None
    track_id: int | None = None

    def compute_similarity(self, other: Face) -> float:
        """Compute cosine similarity with another face.

        Raises:
            ValueError: If either face has no embedding.
        """
        if self.embedding is None or other.embedding is None:
            raise ValueError('Both faces must have embeddings for similarity computation')
        return float(compute_similarity(self.embedding, other.embedding))

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_json(self, indent: int | None = None) -> str:
        """Serialize the Face object to a JSON string.

        Converts numpy arrays to Python lists and numpy scalar types to
        native Python types so the result is fully JSON-serializable.

        Args:
            indent: Number of spaces for pretty-printing. None for compact output.

        Returns:
            JSON string representation of this Face object.

        Example:
            >>> face = faces[0]
            >>> print(face.to_json(indent=2))
        """
        data: dict = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, np.floating):
                value = float(value)
            elif isinstance(value, np.integer):
                value = int(value)
            data[f.name] = value
        return json.dumps(data, indent=indent)

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
        if self.track_id is not None:
            parts.append(f'track_id={self.track_id}')
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
        # Gated per field: a predictor may fill only some of the five, and __repr__ is
        # what you reach for when something has already gone wrong.
        states = [
            f'{name}={value:.2f}'
            for name in ('left_eye_open', 'right_eye_open', 'eyeglasses', 'mask', 'sunglasses')
            if (value := getattr(self, name)) is not None
        ]
        if states:
            parts.append(', '.join(states))
        if self.quality is not None:
            parts.append(f'quality={self.quality:.4f}')
        if self.embedding is not None:
            parts.append(f'embedding_dim={self.embedding.shape[-1]}')
        return ', '.join(parts) + ')'
