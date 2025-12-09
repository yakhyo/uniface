# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo
#
# MLX Base Recognizer Implementation

"""
Base face recognizer class for MLX backend.

This module provides the abstract base class for all MLX-based face
recognition models, implementing common functionality like preprocessing,
face alignment, and embedding extraction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Union

import cv2
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from uniface.face_utils import face_alignment
from uniface.mlx_utils import synchronize, to_numpy


@dataclass
class PreprocessConfig:
    """
    Configuration for preprocessing images before feeding them into the model.
    """

    input_mean: Union[float, List[float]] = 127.5
    input_std: Union[float, List[float]] = 127.5
    input_size: Tuple[int, int] = (112, 112)


class BaseRecognizerMLX(ABC):
    """
    Abstract Base Class for all MLX-based face recognition models.

    This provides the core functionality for preprocessing, inference,
    and embedding extraction using the MLX backend for Apple Silicon.
    """

    def __init__(self, preprocessing: PreprocessConfig) -> None:
        """
        Initialize the recognizer with preprocessing configuration.

        Args:
            preprocessing: Configuration for image preprocessing.
        """
        self.input_mean = preprocessing.input_mean
        self.input_std = preprocessing.input_std
        self.input_size = preprocessing.input_size

        # Subclasses must set this
        self.model: nn.Module = None

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """Build the MLX model. Must be implemented by subclasses."""
        pass

    def preprocess(self, face_img: np.ndarray) -> mx.array:
        """
        Preprocess the image: resize, normalize, and convert to MLX array.

        Args:
            face_img: Input image in BGR format.

        Returns:
            Preprocessed MLX array ready for inference (NHWC format).
        """
        # Resize image
        resized_img = cv2.resize(face_img, self.input_size)

        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Normalize
        if isinstance(self.input_std, (list, tuple)):
            mean = np.array(self.input_mean, dtype=np.float32)
            std = np.array(self.input_std, dtype=np.float32)
        else:
            mean = self.input_mean
            std = self.input_std

        normalized_img = (rgb_img - mean) / std

        # Add batch dimension (H, W, C) -> (1, H, W, C) for NHWC format
        batch_img = np.expand_dims(normalized_img, axis=0)

        return mx.array(batch_img)

    def inference(self, input_tensor: mx.array) -> mx.array:
        """
        Perform MLX inference.

        Args:
            input_tensor: Preprocessed input tensor in NHWC format.

        Returns:
            Embedding tensor.
        """
        embedding = self.model(input_tensor)
        synchronize(embedding)
        return embedding

    def get_embedding(self, image: np.ndarray, landmarks: np.ndarray = None) -> np.ndarray:
        """
        Extract face embedding from an image.

        Args:
            image: Input face image (BGR format).
            landmarks: Facial landmarks (5 points) for alignment. Optional.

        Returns:
            Face embedding vector (typically 512-dimensional).
        """
        # Align face if landmarks provided
        if landmarks is not None:
            aligned_face, _ = face_alignment(image, landmarks, image_size=self.input_size)
        else:
            aligned_face = image

        # Preprocess
        input_tensor = self.preprocess(aligned_face)

        # Inference
        embedding = self.inference(input_tensor)

        # Convert to numpy
        return to_numpy(embedding)

    def get_normalized_embedding(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract L2-normalized face embedding from an image.

        Args:
            image: Input face image (BGR format).
            landmarks: Facial landmarks (5 points) for alignment.

        Returns:
            Normalized face embedding vector (typically 512-dimensional).
        """
        embedding = self.get_embedding(image, landmarks)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding
