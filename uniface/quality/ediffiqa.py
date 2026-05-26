# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import cv2
import numpy as np

from uniface.constants import EDifFIQAWeights
from uniface.face_utils import face_alignment
from uniface.log import Logger
from uniface.model_store import verify_model_weights
from uniface.onnx_utils import create_onnx_session
from uniface.types import QualityResult

from .base import BaseQualityEstimator

__all__ = ['EDifFIQA']


class EDifFIQA(BaseQualityEstimator):
    """
    eDifFIQA: Face Image Quality Assessment with ONNX Runtime.

    Predicts a single scalar quality score from an aligned 112x112 face crop.
    Higher score = better quality. Supports four backbones via the
    `EDifFIQAWeights` enum (T/S/M/L).

    Paper:
        Babnik et al., "eDifFIQA: Towards Efficient Face Image Quality
        Assessment based on Denoising Diffusion Probabilistic Models",
        IEEE T-BIOM 2024. https://ieeexplore.ieee.org/document/10468647

    Code:
        https://github.com/yakhyo/face-image-quality-assessment

    Args:
        model_name: The enum specifying the model variant to load.
            Defaults to `EDifFIQAWeights.T` (smallest, ~6.6 MB).
        providers: ONNX Runtime execution providers. If None, auto-detects
            the best available provider.

    Example:
        >>> from uniface.detection import SCRFD
        >>> from uniface.quality import EDifFIQA
        >>>
        >>> detector = SCRFD()
        >>> quality = EDifFIQA()
        >>>
        >>> faces = detector.detect(image)
        >>> for face in faces:
        ...     result = quality.predict(image, face.landmarks)
        ...     print(f'Quality: {result.score:.4f}')
    """

    def __init__(
        self,
        model_name: EDifFIQAWeights = EDifFIQAWeights.T,
        providers: list[str] | None = None,
    ) -> None:
        Logger.info(f'Initializing EDifFIQA with model={model_name.name}')

        self.providers = providers
        self.model_path = verify_model_weights(model_name)

        # ((image / 255) - 0.5) / 0.5  ==  (image - 127.5) / 127.5
        self.normalization_mean = 127.5
        self.normalization_std = 127.5

        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Initialize the ONNX model from the stored model path.

        Raises:
            RuntimeError: If the model fails to load or initialize.
        """
        try:
            self.session = create_onnx_session(self.model_path, providers=self.providers)

            input_cfg = self.session.get_inputs()[0]
            self.input_name = input_cfg.name
            # Input shape is (batch, channels, height, width) - we need (width, height)
            self.input_size = tuple(input_cfg.shape[2:4][::-1])

            self.output_name = self.session.get_outputs()[0].name

            Logger.info(f'EDifFIQA initialized with input size {self.input_size}')

        except Exception as e:
            Logger.error(f"Failed to load EDifFIQA model from '{self.model_path}'", exc_info=True)
            raise RuntimeError(f'Failed to initialize EDifFIQA model: {e}') from e

    def preprocess(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Preprocess an aligned face crop for model inference.

        Converts BGR -> RGB, normalizes with mean=127.5/std=127.5, and
        reshapes to NCHW.

        Args:
            aligned_face: Aligned face crop in BGR format.

        Returns:
            Preprocessed image tensor with shape (1, 3, H, W).
        """
        return cv2.dnn.blobFromImage(
            aligned_face,
            scalefactor=1.0 / self.normalization_std,
            size=self.input_size,
            mean=(self.normalization_mean,) * 3,
            swapRB=True,
        )

    def score_aligned(self, aligned_face: np.ndarray) -> QualityResult:
        """
        Score a pre-aligned face crop.

        Args:
            aligned_face: Aligned face crop in BGR format.

        Returns:
            QualityResult with the predicted score.
        """
        blob = self.preprocess(aligned_face)
        output = self.session.run([self.output_name], {self.input_name: blob})[0]
        return QualityResult(score=float(np.squeeze(output)))

    def predict(self, image: np.ndarray, landmarks: np.ndarray) -> QualityResult:
        """
        Align the face using 5-point landmarks, then score it.

        Args:
            image: Input image in BGR format containing the face.
            landmarks: (5, 2) array of 5-point facial landmarks.

        Returns:
            QualityResult with the predicted score.
        """
        aligned, _ = face_alignment(image, landmarks.astype(np.float32))
        return self.score_aligned(aligned)
