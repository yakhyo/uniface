# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo
#
# MLX Emotion Recognition Implementation

"""
Emotion recognition model implemented in MLX for Apple Silicon.

This module provides an MLX implementation of the DDAMFN (Dual Direction
Attention Mixed Feature Network) emotion recognition model.

Supported emotions:
- Neutral, Happy, Sad, Surprise, Fear, Disgust, Angry
- Contempt (optional, for 8-class model)
"""

from typing import List, Tuple, Union

import cv2
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from uniface.attribute.base import Attribute
from uniface.constants import DDAMFNWeights
from uniface.face_utils import face_alignment
from uniface.log import Logger
from uniface.mlx_utils import load_mlx_weights, synchronize, to_numpy
from uniface.model_store import get_weights_path
from uniface.nn.conv import ConvBNReLU, DepthwiseSeparableConv

__all__ = ['EmotionMLX']


class ChannelAttention(nn.Module):
    """
    Channel attention module (Squeeze-and-Excitation style).

    Applies attention weights to each channel based on global features.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Input tensor (N, H, W, C) in NHWC format.

        Returns:
            Channel-attended output.
        """
        # Global average pooling
        avg_pool = mx.mean(x, axis=(1, 2), keepdims=True)  # (N, 1, 1, C)

        # Squeeze-Excitation
        weight = nn.relu(self.fc1(avg_pool))
        weight = mx.sigmoid(self.fc2(weight))  # (N, 1, 1, C)

        return x * weight


class SpatialAttention(nn.Module):
    """
    Spatial attention module.

    Applies attention weights to each spatial location based on
    channel-wise statistics.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Input tensor (N, H, W, C) in NHWC format.

        Returns:
            Spatially-attended output.
        """
        # Channel-wise max and mean
        max_pool = mx.max(x, axis=-1, keepdims=True)  # (N, H, W, 1)
        avg_pool = mx.mean(x, axis=-1, keepdims=True)  # (N, H, W, 1)

        # Concatenate and apply conv
        combined = mx.concatenate([max_pool, avg_pool], axis=-1)  # (N, H, W, 2)
        weight = mx.sigmoid(self.conv(combined))  # (N, H, W, 1)

        return x * weight


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Combines channel and spatial attention for feature refinement.
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class DDAMFNBlock(nn.Module):
    """
    DDAMFN building block with attention mechanism.

    Uses depthwise separable convolutions with CBAM attention.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, stride=stride)
        self.attention = CBAM(out_channels)
        self.use_residual = stride == 1 and in_channels == out_channels

    def __call__(self, x: mx.array) -> mx.array:
        identity = x
        out = self.conv(x)
        out = self.attention(out)
        if self.use_residual:
            out = out + identity
        return out


class DDAMFNBackbone(nn.Module):
    """
    DDAMFN (Dual Direction Attention Mixed Feature Network) backbone.

    This is a simplified implementation of DDAMFN for emotion recognition,
    using attention mechanisms to focus on discriminative facial features.

    Args:
        num_classes: Number of emotion classes (7 or 8).
        input_size: Input image size. Default: (112, 112).
    """

    def __init__(
        self,
        num_classes: int = 7,
        input_size: Tuple[int, int] = (112, 112),
        width_mult: float = 1.0,
    ):
        super().__init__()

        def ch(c: int) -> int:
            return max(int(c * width_mult), 8)

        # Stem
        self.stem = nn.Sequential(
            ConvBNReLU(3, ch(32), kernel_size=3, stride=2, padding=1),
            ConvBNReLU(ch(32), ch(32), kernel_size=3, stride=1, padding=1),
        )

        # Feature extraction stages with attention
        self.stage1 = nn.Sequential(
            DDAMFNBlock(ch(32), ch(64), stride=2),
            DDAMFNBlock(ch(64), ch(64), stride=1),
        )

        self.stage2 = nn.Sequential(
            DDAMFNBlock(ch(64), ch(128), stride=2),
            DDAMFNBlock(ch(128), ch(128), stride=1),
        )

        self.stage3 = nn.Sequential(
            DDAMFNBlock(ch(128), ch(256), stride=2),
            DDAMFNBlock(ch(256), ch(256), stride=1),
        )

        self.stage4 = nn.Sequential(
            DDAMFNBlock(ch(256), ch(512), stride=2),
            DDAMFNBlock(ch(512), ch(512), stride=1),
        )

        # Final classifier
        self.fc = nn.Linear(ch(512), num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor (N, H, W, 3) in NHWC format.

        Returns:
            Logits of shape (N, num_classes).
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Global average pooling
        x = mx.mean(x, axis=(1, 2))  # (N, C)

        # Classify
        x = self.fc(x)

        return x


class EmotionMLX(Attribute):
    """
    Emotion recognition model using MLX backend.

    Predicts one of 7 or 8 emotion categories:
    - Neutral, Happy, Sad, Surprise, Fear, Disgust, Angry
    - Contempt (for 8-class model only)

    Args:
        model_weights: DDAMFN model weights to use.
        input_size: Input image size. Default: (112, 112).
    """

    def __init__(
        self,
        model_weights: DDAMFNWeights = DDAMFNWeights.AFFECNET7,
        input_size: Tuple[int, int] = (112, 112),
    ) -> None:
        Logger.info(f'Initializing Emotion (MLX) with model={model_weights.name}')

        self.input_size = input_size

        # Define emotion labels based on model
        self.emotion_labels = [
            'Neutral',
            'Happy',
            'Sad',
            'Surprise',
            'Fear',
            'Disgust',
            'Angry',
        ]
        if model_weights == DDAMFNWeights.AFFECNET8:
            self.emotion_labels.append('Contempt')

        num_classes = len(self.emotion_labels)

        # Build model
        self.model = DDAMFNBackbone(
            num_classes=num_classes,
            input_size=input_size,
            width_mult=1.0,
        )

        # Load weights
        try:
            weights_path = get_weights_path(model_weights, backend='mlx')
            load_mlx_weights(self.model, weights_path)
            Logger.info(f'Loaded MLX weights from {weights_path}')
        except (ValueError, FileNotFoundError) as e:
            Logger.warning(f'MLX weights not available: {e}. Model initialized without weights.')

        # Set to inference mode
        self.model.train(False)

        # ImageNet normalization (used by DDAMFN)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, image: np.ndarray, landmark: Union[List, np.ndarray]) -> mx.array:
        """
        Align face using landmarks and prepare for inference.

        Args:
            image: Full input image in BGR format.
            landmark: 5-point facial landmarks.

        Returns:
            Preprocessed MLX array.
        """
        landmark = np.asarray(landmark)

        # Align face
        aligned_image, _ = face_alignment(image, landmark, image_size=self.input_size)

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Resize if needed
        if rgb_image.shape[:2] != self.input_size:
            rgb_image = cv2.resize(rgb_image, self.input_size)

        # Normalize (ImageNet stats)
        normalized_image = (rgb_image / 255.0 - self.mean) / self.std

        # Add batch dimension (H, W, C) -> (1, H, W, C)
        batch_image = np.expand_dims(normalized_image, axis=0)

        return mx.array(batch_image)

    def inference(self, input_tensor: mx.array) -> mx.array:
        """Perform MLX inference."""
        output = self.model(input_tensor)
        synchronize(output)
        return output

    def postprocess(self, prediction: np.ndarray) -> Tuple[str, float]:
        """
        Process raw output to get emotion label and confidence.

        Args:
            prediction: Raw logits from model.

        Returns:
            Tuple of (emotion_label, confidence).
        """
        # Softmax
        exp_pred = np.exp(prediction - np.max(prediction))
        probabilities = exp_pred / np.sum(exp_pred)

        # Get prediction
        pred_index = np.argmax(probabilities)
        emotion_label = self.emotion_labels[pred_index]
        confidence = float(probabilities[pred_index])

        return emotion_label, confidence

    def predict(self, image: np.ndarray, landmark: Union[List, np.ndarray]) -> Tuple[str, float]:
        """
        Predict emotion for a face.

        Args:
            image: Full input image in BGR format.
            landmark: 5-point facial landmarks.

        Returns:
            Tuple of (emotion_label, confidence).
        """
        # Preprocess
        input_tensor = self.preprocess(image, landmark)

        # Inference
        mlx_output = self.inference(input_tensor)

        # Convert to numpy
        prediction = to_numpy(mlx_output)[0]

        # Postprocess
        emotion_label, confidence = self.postprocess(prediction)

        return emotion_label, confidence
