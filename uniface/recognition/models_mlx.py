# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo
#
# MLX Recognition Models Implementation

"""
Face recognition models implemented in MLX for Apple Silicon.

This module provides MLX implementations of popular face recognition models:
- ArcFace: State-of-the-art face recognition with additive angular margin loss
- MobileFace: Lightweight MobileNet-based face recognition
- SphereFace: Angular softmax-based face recognition

All models output 512-dimensional face embeddings.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from uniface.constants import ArcFaceWeights, MobileFaceWeights, SphereFaceWeights
from uniface.log import Logger
from uniface.mlx_utils import load_mlx_weights
from uniface.model_store import get_weights_path
from uniface.nn.conv import ConvBN, ConvBNReLU, InvertedResidual

from .base_mlx import BaseRecognizerMLX, PreprocessConfig

__all__ = ['ArcFaceMLX', 'MobileFaceMLX', 'SphereFaceMLX']


class GlobalDepthwiseConv(nn.Module):
    """
    Global Depthwise Convolution block for face recognition.

    This is used in MobileFaceNet-style architectures to extract global features
    before the final embedding layer.
    """

    def __init__(self, in_channels: int, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn = nn.BatchNorm(in_channels)

    def __call__(self, x: mx.array) -> mx.array:
        return self.bn(self.conv(x))


class MobileFaceNetBackbone(nn.Module):
    """
    MobileFaceNet backbone for face recognition.

    A compact MobileNetV2-style architecture optimized for face recognition.
    Uses inverted residuals with global depthwise convolution for embedding extraction.

    Architecture:
    - Stem: Conv(3x3, s=2) + GDConv(3x3)
    - Stages: Inverted residual blocks with varying expansion ratios
    - Head: GDConv(7x7) + Linear(512)

    Output: 512-dimensional embedding
    """

    def __init__(self, embedding_dim: int = 512, width_mult: float = 1.0):
        super().__init__()

        def ch(c: int) -> int:
            return max(int(c * width_mult), 8)

        # Stem
        self.stem = nn.Sequential(
            ConvBNReLU(3, ch(64), kernel_size=3, stride=2, padding=1),
            # Depthwise separable
            nn.Conv2d(ch(64), ch(64), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm(ch(64)),
        )

        # MobileFaceNet configuration: (t, c, n, s)
        # t = expansion ratio, c = output channels, n = repeat, s = stride
        inverted_residual_setting = [
            [2, 64, 5, 2],  # Stage 1
            [4, 128, 1, 2],  # Stage 2
            [2, 128, 6, 1],  # Stage 3
            [4, 128, 1, 2],  # Stage 4
            [2, 128, 2, 1],  # Stage 5
        ]

        # Build stages
        self.stages = []
        in_channels = ch(64)

        for stage_idx, (t, c, n, s) in enumerate(inverted_residual_setting):
            out_channels = ch(c)
            layers = []
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(
                    InvertedResidual(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride,
                        expand_ratio=t,
                    )
                )
                in_channels = out_channels
            self.stages.append(nn.Sequential(*layers))

        # Final conv before embedding
        self.conv_final = ConvBNReLU(ch(128), ch(512), kernel_size=1)

        # Global depthwise conv (7x7 for 112x112 input -> 7x7 feature map)
        self.gdconv = GlobalDepthwiseConv(ch(512), kernel_size=7)

        # Embedding layer
        self.fc = nn.Linear(ch(512), embedding_dim, bias=False)
        self.bn_fc = nn.BatchNorm(embedding_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor (N, H, W, 3) in NHWC format, typically (N, 112, 112, 3).

        Returns:
            Embedding tensor of shape (N, embedding_dim).
        """
        # Stem
        x = self.stem(x)
        x = nn.relu(x)

        # Stages
        for stage in self.stages:
            x = stage(x)

        # Final conv
        x = self.conv_final(x)

        # Global depthwise conv
        x = self.gdconv(x)

        # Flatten and embed
        x = x.reshape(x.shape[0], -1)  # (N, C)
        x = self.fc(x)
        x = self.bn_fc(x)

        return x


class ArcFaceBackbone(nn.Module):
    """
    ArcFace backbone using MobileNet-style architecture.

    This is a simplified implementation that uses MobileNetV2 as the backbone
    with appropriate modifications for face recognition.
    """

    def __init__(self, embedding_dim: int = 512, width_mult: float = 1.0):
        super().__init__()

        # Use MobileFaceNet architecture (proven effective for face recognition)
        self.backbone = MobileFaceNetBackbone(embedding_dim=embedding_dim, width_mult=width_mult)

    def __call__(self, x: mx.array) -> mx.array:
        return self.backbone(x)


class SphereFaceBackbone(nn.Module):
    """
    SphereFace backbone (simplified).

    Original SphereFace uses a custom CNN architecture with PReLU activations.
    This implementation uses a similar structure adapted for MLX.

    The key insight of SphereFace is the angular softmax loss, but for inference
    we just need the feature extraction backbone.
    """

    def __init__(self, embedding_dim: int = 512, num_layers: int = 20):
        super().__init__()

        # SphereFace-20 configuration
        if num_layers == 20:
            units = [1, 2, 4, 1]
        elif num_layers == 36:
            units = [2, 4, 8, 2]
        else:
            units = [1, 2, 4, 1]  # Default to 20

        # Initial convolution
        self.conv1 = ConvBN(3, 64, kernel_size=3, stride=2, padding=1)
        self.prelu1 = nn.PReLU()

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, units[0])
        self.layer2 = self._make_layer(64, 128, units[1], stride=2)
        self.layer3 = self._make_layer(128, 256, units[2], stride=2)
        self.layer4 = self._make_layer(256, 512, units[3], stride=2)

        # Embedding
        self.fc = nn.Linear(512 * 7 * 7, embedding_dim, bias=False)
        self.bn_fc = nn.BatchNorm(embedding_dim)

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1) -> nn.Module:
        """Create a residual layer."""
        layers = []

        # First block with potential stride
        layers.append(ConvBN(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.PReLU())
        layers.append(ConvBN(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.PReLU())

        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ConvBN(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.PReLU())
            layers.append(ConvBN(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.PReLU())

        return nn.Sequential(*layers)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor (N, H, W, 3) in NHWC format.

        Returns:
            Embedding tensor of shape (N, embedding_dim).
        """
        x = self.prelu1(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Flatten and embed
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.bn_fc(x)

        return x


class ArcFaceMLX(BaseRecognizerMLX):
    """
    ArcFace face recognition model using MLX backend.

    ArcFace uses Additive Angular Margin Loss to enhance discriminative
    power of face recognition models. This implementation provides
    inference-only capability on Apple Silicon.

    Args:
        model_name: ArcFace model variant to use.
        preprocessing: Optional custom preprocessing configuration.
    """

    def __init__(
        self,
        model_name: ArcFaceWeights = ArcFaceWeights.MNET,
        preprocessing: Optional[PreprocessConfig] = None,
    ) -> None:
        if preprocessing is None:
            preprocessing = PreprocessConfig(input_mean=127.5, input_std=127.5, input_size=(112, 112))

        super().__init__(preprocessing=preprocessing)

        self.model_name = model_name

        Logger.info(f'Initializing ArcFace (MLX) with model={model_name}')

        # Build model
        self.model = self._build_model()

        # Load weights
        try:
            weights_path = get_weights_path(model_name, backend='mlx')
            load_mlx_weights(self.model, weights_path)
            Logger.info(f'Loaded MLX weights from {weights_path}')
        except (ValueError, FileNotFoundError) as e:
            Logger.warning(f'MLX weights not available: {e}. Model initialized without weights.')

        # Set to inference mode
        self.model.train(False)

        # Compile for better performance
        self._compile_model()
        Logger.debug('Compiled recognition model with mx.compile')

    def _build_model(self) -> nn.Module:
        """Build the ArcFace model."""
        # Determine width multiplier from model name
        width_mult = 1.0
        if 'mnet' in self.model_name.value.lower():
            width_mult = 1.0

        return ArcFaceBackbone(embedding_dim=512, width_mult=width_mult)


class MobileFaceMLX(BaseRecognizerMLX):
    """
    MobileFace face recognition model using MLX backend.

    MobileFaceNet is a lightweight face recognition model designed
    for mobile and embedded devices.

    Args:
        model_name: MobileFace model variant to use.
        preprocessing: Optional custom preprocessing configuration.
    """

    def __init__(
        self,
        model_name: MobileFaceWeights = MobileFaceWeights.MNET_V2,
        preprocessing: Optional[PreprocessConfig] = None,
    ) -> None:
        if preprocessing is None:
            preprocessing = PreprocessConfig(input_mean=127.5, input_std=127.5, input_size=(112, 112))

        super().__init__(preprocessing=preprocessing)

        self.model_name = model_name

        Logger.info(f'Initializing MobileFace (MLX) with model={model_name}')

        # Build model
        self.model = self._build_model()

        # Load weights
        try:
            weights_path = get_weights_path(model_name, backend='mlx')
            load_mlx_weights(self.model, weights_path)
            Logger.info(f'Loaded MLX weights from {weights_path}')
        except (ValueError, FileNotFoundError) as e:
            Logger.warning(f'MLX weights not available: {e}. Model initialized without weights.')

        # Set to inference mode
        self.model.train(False)

        # Compile for better performance
        self._compile_model()
        Logger.debug('Compiled recognition model with mx.compile')

    def _build_model(self) -> nn.Module:
        """Build the MobileFace model."""
        # Determine width multiplier from model name
        width_mult = 1.0
        model_name_lower = self.model_name.value.lower()
        if 'v1' in model_name_lower:
            width_mult = 1.0
        elif 'v2' in model_name_lower:
            width_mult = 1.0
        elif 'v3' in model_name_lower:
            width_mult = 1.0

        return MobileFaceNetBackbone(embedding_dim=512, width_mult=width_mult)


class SphereFaceMLX(BaseRecognizerMLX):
    """
    SphereFace face recognition model using MLX backend.

    SphereFace was one of the first to introduce angular margin loss
    for face recognition. This implementation provides the inference
    backbone.

    Args:
        model_name: SphereFace model variant to use.
        preprocessing: Optional custom preprocessing configuration.
    """

    def __init__(
        self,
        model_name: SphereFaceWeights = SphereFaceWeights.SPHERE20,
        preprocessing: Optional[PreprocessConfig] = None,
    ) -> None:
        if preprocessing is None:
            preprocessing = PreprocessConfig(input_mean=127.5, input_std=127.5, input_size=(112, 112))

        super().__init__(preprocessing=preprocessing)

        self.model_name = model_name

        Logger.info(f'Initializing SphereFace (MLX) with model={model_name}')

        # Build model
        self.model = self._build_model()

        # Load weights
        try:
            weights_path = get_weights_path(model_name, backend='mlx')
            load_mlx_weights(self.model, weights_path)
            Logger.info(f'Loaded MLX weights from {weights_path}')
        except (ValueError, FileNotFoundError) as e:
            Logger.warning(f'MLX weights not available: {e}. Model initialized without weights.')

        # Set to inference mode
        self.model.train(False)

        # Compile for better performance
        self._compile_model()
        Logger.debug('Compiled recognition model with mx.compile')

    def _build_model(self) -> nn.Module:
        """Build the SphereFace model."""
        # Determine number of layers from model name
        num_layers = 20
        if 'sphere20' in self.model_name.value.lower():
            num_layers = 20
        elif 'sphere36' in self.model_name.value.lower():
            num_layers = 36

        return SphereFaceBackbone(embedding_dim=512, num_layers=num_layers)
