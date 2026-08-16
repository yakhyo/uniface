# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np

from uniface.types import Face

__all__ = ['BaseDetector']


class BaseDetector(ABC):
    """Abstract base class for all face detectors.

    This class defines the interface that all face detectors must implement,
    ensuring consistency across different detection methods.

    Attributes:
        config: Dictionary containing detector configuration parameters.
        supports_landmarks: Whether the detector emits facial landmarks alongside boxes.
            Opt-in: defaults to False so a subclass that only produces boxes is never
            mistaken for a landmark detector. False means `Face.landmarks` is empty and
            landmark-driven stages are unavailable.
        supports_alignment: Whether those landmarks fit the 5-point alignment template
            (left eye, right eye, nose, left mouth corner, right mouth corner) in
            `uniface.face_utils.reference_alignment`. False means recognition, quality
            scoring, and XSeg parsing cannot consume them. Also opt-in: a boxes-only
            subclass that declares neither flag is safe by default, and setting this to
            True without `supports_landmarks = True` is meaningless.
    """

    supports_landmarks: bool = False
    supports_alignment: bool = False

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the detector with configuration parameters.

        Args:
            **kwargs: Detector-specific configuration parameters.
        """
        self.config: dict[str, Any] = kwargs

    @abstractmethod
    def detect(self, image: np.ndarray, **kwargs: Any) -> list[Face]:
        """Detect faces in an image.

        Args:
            image: Input image as numpy array with shape (H, W, C) in BGR format.
            **kwargs: Additional detection parameters.

        Returns:
            List of detected Face objects, each containing:
                - bbox: Bounding box coordinates with shape (4,) as [x1, y1, x2, y2].
                - confidence: Detection confidence score (0.0 to 1.0).
                - landmarks: Facial landmarks with shape (K, 2). K is 5 for detectors
                  whose `supports_alignment` is True; BlazeFace emits 6.

        Example:
            >>> faces = detector.detect(image)
            >>> for face in faces:
            ...     bbox = face.bbox  # np.ndarray with shape (4,)
            ...     confidence = face.confidence  # float
            ...     landmarks = face.landmarks  # np.ndarray with shape (K, 2)
        """

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess input image for detection.

        Args:
            image: Input image with shape (H, W, C).

        Returns:
            Preprocessed image tensor ready for inference.
        """

    @abstractmethod
    def postprocess(self, outputs: Any, **kwargs: Any) -> Any:
        """Postprocess model outputs to get final detections.

        Args:
            outputs: Raw model outputs.
            **kwargs: Additional postprocessing parameters.

        Returns:
            Processed outputs (implementation-specific format, typically tuple of arrays).
        """

    def __str__(self) -> str:
        """String representation of the detector."""
        return f'{self.__class__.__name__}({self.config})'

    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()

    def get_info(self) -> dict[str, Any]:
        """Get detector information and configuration.

        Returns:
            Dictionary containing detector name, landmark support, alignment support, and config.
        """
        return {
            'name': self.__class__.__name__,
            'supports_landmarks': self.supports_landmarks,
            'supports_alignment': self.supports_alignment,
            'config': self.config,
        }

    def __call__(self, image: np.ndarray, **kwargs: Any) -> list[Face]:
        """Callable shortcut for the `detect` method.

        Args:
            image: Input image as numpy array with shape (H, W, C) in BGR format.
            **kwargs: Additional detection parameters.

        Returns:
            List of detected Face objects.
        """
        return self.detect(image, **kwargs)

    def _select_top_detections(
        self,
        detections: np.ndarray,
        landmarks: np.ndarray,
        max_num: int,
        original_shape: tuple[int, int],
        metric: Literal['default', 'max'] = 'max',
        center_weight: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter detections to keep only top max_num faces.

        Ranks faces by area and/or distance from image center, then selects
        the top max_num detections.

        Args:
            detections: Array of shape (N, 5) as [x1, y1, x2, y2, confidence].
            landmarks: Array of shape (N, K, 2). Only axis 0 is indexed, so any K works.
            max_num: Maximum number of faces to keep. If 0 or >= N, returns all.
            original_shape: Original image shape as (height, width).
            metric: Ranking metric:
                - 'max': Rank by bounding box area only.
                - 'default': Rank by area minus center distance penalty.
            center_weight: Weight for center distance penalty (only used with 'default' metric).

        Returns:
            Filtered (detections, landmarks) tuple with at most max_num faces.
        """
        if max_num <= 0 or detections.shape[0] <= max_num:
            return detections, landmarks

        # Calculate bounding box areas
        area = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])

        # Calculate offsets from image center
        center_y, center_x = original_shape[0] // 2, original_shape[1] // 2
        offsets = np.vstack(
            [
                (detections[:, 0] + detections[:, 2]) / 2 - center_x,
                (detections[:, 1] + detections[:, 3]) / 2 - center_y,
            ]
        )
        offset_dist_squared = np.sum(np.power(offsets, 2.0), axis=0)

        # Calculate ranking scores based on metric
        if metric == 'max':
            scores = area
        else:
            scores = area - offset_dist_squared * center_weight

        # Select top max_num by score
        top_indices = np.argsort(scores)[::-1][:max_num]
        return detections[top_indices], landmarks[top_indices]

    @staticmethod
    def _detections_to_faces(detections: np.ndarray, landmarks: np.ndarray) -> list[Face]:
        """Convert detection arrays to Face objects.

        Args:
            detections: Array of shape (N, 5) as [x1, y1, x2, y2, confidence].
            landmarks: Array of shape (N, K, 2). Only axis 0 is indexed, so any K works.

        Returns:
            List of Face objects.
        """
        faces = []
        for i in range(detections.shape[0]):
            face = Face(
                bbox=detections[i, :4],
                confidence=float(detections[i, 4]),
                landmarks=landmarks[i],
            )
            faces.append(face)
        return faces
