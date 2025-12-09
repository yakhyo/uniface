# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo
#
# MLX Landmark106 Implementation

"""
106-point facial landmark detector implemented in MLX for Apple Silicon.

This module provides MLX implementation of the 106-point landmark detector,
which predicts detailed facial keypoints including face contour, eyebrows,
eyes, nose, and mouth regions.
"""

from typing import Tuple

import cv2
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from uniface.constants import LandmarkWeights
from uniface.face_utils import bbox_center_alignment, transform_points_2d
from uniface.log import Logger
from uniface.mlx_utils import load_mlx_weights, synchronize, to_numpy
from uniface.model_store import get_weights_path
from uniface.nn.conv import ConvBNReLU, DepthwiseSeparableConv

from .base import BaseLandmarker

__all__ = ['Landmark106MLX']


class LandmarkBackbone(nn.Module):
    """
    Lightweight backbone for facial landmark detection.

    Uses a compact MobileNet-style architecture with depthwise separable
    convolutions for efficient inference.

    Architecture:
    - Stem: Conv(3x3, s=2) -> DepthwiseSeparable blocks
    - Features progressively downsample spatial dimensions
    - Final conv produces landmark coordinates

    Args:
        input_size: Input image size (width, height). Default: (192, 192).
        num_landmarks: Number of landmark points to predict. Default: 106.
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (192, 192),
        num_landmarks: int = 106,
        width_mult: float = 0.5,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_landmarks = num_landmarks

        def ch(c: int) -> int:
            return max(int(c * width_mult), 8)

        # Stem
        self.stem = ConvBNReLU(3, ch(64), kernel_size=3, stride=2, padding=1)

        # Stages - similar to MobileNetV1 but simplified
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(ch(64), ch(64), stride=1),
            DepthwiseSeparableConv(ch(64), ch(128), stride=2),
        )

        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(ch(128), ch(128), stride=1),
            DepthwiseSeparableConv(ch(128), ch(256), stride=2),
        )

        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(ch(256), ch(256), stride=1),
            DepthwiseSeparableConv(ch(256), ch(512), stride=2),
        )

        self.stage4 = nn.Sequential(
            DepthwiseSeparableConv(ch(512), ch(512), stride=1),
            DepthwiseSeparableConv(ch(512), ch(512), stride=2),
        )

        # Final conv to reduce channels
        self.conv_final = ConvBNReLU(ch(512), ch(256), kernel_size=1)

        # Global average pooling will reduce to (N, 1, 1, C)
        # Then fully connected to landmark coordinates
        self.fc = nn.Linear(ch(256), num_landmarks * 2)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor (N, H, W, 3) in NHWC format.

        Returns:
            Landmark coordinates of shape (N, num_landmarks * 2).
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv_final(x)

        # Global average pooling
        x = mx.mean(x, axis=(1, 2))  # (N, C)

        # Predict landmarks
        x = self.fc(x)

        return x


class Landmark106MLX(BaseLandmarker):
    """
    106-point facial landmark detector using MLX backend.

    This model predicts 106 facial keypoints that cover:
    - Face contour (33 points)
    - Left/Right eyebrows (8+8 points)
    - Left/Right eyes (8+8 points)
    - Nose (13 points)
    - Upper/Lower lips (22+6 points)

    Args:
        model_name: Landmark model weights to use.
        input_size: Input image size. Default: (192, 192).
    """

    def __init__(
        self,
        model_name: LandmarkWeights = LandmarkWeights.DEFAULT,
        input_size: Tuple[int, int] = (192, 192),
    ) -> None:
        Logger.info(f'Initializing Landmark106 (MLX) with model={model_name}, input_size={input_size}')

        self.input_size = input_size
        self.input_std = 1.0
        self.input_mean = 0.0
        self.lmk_dim = 2
        self.lmk_num = 106

        # Build model
        self.model = LandmarkBackbone(
            input_size=input_size,
            num_landmarks=106,
            width_mult=0.5,
        )

        # Load weights
        try:
            weights_path = get_weights_path(model_name, backend='mlx')
            load_mlx_weights(self.model, weights_path)
            Logger.info(f'Loaded MLX weights from {weights_path}')
        except (ValueError, FileNotFoundError) as e:
            Logger.warning(f'MLX weights not available: {e}. Model initialized without weights.')

        # Set to inference mode
        self.model.train(False)

    def preprocess(self, image: np.ndarray, bbox: np.ndarray) -> Tuple[mx.array, np.ndarray]:
        """
        Prepare a face crop for inference.

        Args:
            image: Full source image in BGR format.
            bbox: Face bounding box [x1, y1, x2, y2].

        Returns:
            Tuple of (preprocessed MLX array, transformation matrix).
        """
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        scale = self.input_size[0] / (max(width, height) * 1.5)

        aligned_face, transform_matrix = bbox_center_alignment(image, center, self.input_size[0], scale, 0.0)

        # Convert BGR to RGB and normalize
        rgb_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB).astype(np.float32)
        normalized_face = (rgb_face - self.input_mean) / self.input_std

        # Add batch dimension (H, W, C) -> (1, H, W, C)
        batch_face = np.expand_dims(normalized_face, axis=0)

        return mx.array(batch_face), transform_matrix

    def inference(self, input_tensor: mx.array) -> mx.array:
        """Perform MLX inference."""
        output = self.model(input_tensor)
        synchronize(output)
        return output

    def postprocess(self, predictions: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """
        Convert raw model predictions to original image coordinates.

        Args:
            predictions: Raw landmark coordinates from model.
            transform_matrix: Affine transformation from preprocessing.

        Returns:
            Landmarks in original image coordinates with shape (106, 2).
        """
        landmarks = predictions.reshape((-1, 2))

        # Denormalize to input size
        landmarks[:, 0:2] += 1
        landmarks[:, 0:2] *= self.input_size[0] // 2

        # Apply inverse transform
        inverse_matrix = cv2.invertAffineTransform(transform_matrix)
        landmarks = transform_points_2d(landmarks, inverse_matrix)

        return landmarks

    def get_landmarks(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Predict facial landmarks for a face in the image.

        Args:
            image: Full source image in BGR format.
            bbox: Face bounding box [x1, y1, x2, y2].

        Returns:
            Array of predicted landmarks with shape (106, 2).
        """
        # Preprocess
        input_tensor, transform_matrix = self.preprocess(image, bbox)

        # Inference
        mlx_output = self.inference(input_tensor)

        # Convert to numpy
        raw_predictions = to_numpy(mlx_output)[0]

        # Postprocess
        landmarks = self.postprocess(raw_predictions, transform_matrix)

        return landmarks
