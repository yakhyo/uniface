# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo
#
# MLX Backbone Networks

"""
Backbone networks for face detection and recognition.

This module implements MobileNetV1 and MobileNetV2 backbones in MLX,
which are used as feature extractors for RetinaFace, ArcFace, and other models.
"""

from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from uniface.nn.conv import (
    ConvBNReLU,
    ConvBNReLU6,
    ConvReLU6Fused,
    DepthwiseSeparableConv,
    InvertedResidual,
    InvertedResidualFused,
)

__all__ = [
    'MobileNetV1',
    'MobileNetV2',
    'MobileNetV2Fused',
]


class MobileNetV1(nn.Module):
    """
    MobileNetV1 backbone for feature extraction.

    This is a lightweight CNN architecture using depthwise separable convolutions.
    Used as the backbone for RetinaFace (mnet025, mnet050, mnet_v1 variants).

    Args:
        width_mult: Width multiplier (0.25, 0.5, 0.75, 1.0). Default: 1.0.
        return_stages: Which stages to return features from. Default: [2, 3, 4].
            Stage indices: 0=stem, 1=stage1, 2=stage2, 3=stage3, 4=stage4

    Returns:
        List of feature maps from specified stages.
    """

    def __init__(
        self,
        width_mult: float = 1.0,
        return_stages: Optional[List[int]] = None,
    ):
        super().__init__()
        self.width_mult = width_mult
        self.return_stages = return_stages or [2, 3, 4]

        def _make_divisible(v: float, divisor: int = 8) -> int:
            """Round to nearest divisible value."""
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def ch(c: int) -> int:
            """Apply width multiplier to channel count."""
            return _make_divisible(c * width_mult)

        # Stage 0: Stem (stride 2)
        self.stem = ConvBNReLU(3, ch(32), kernel_size=3, stride=2, padding=1)

        # Stage 1: stride 1 -> stride 2 (output stride 4)
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(ch(32), ch(64), stride=1),
            DepthwiseSeparableConv(ch(64), ch(128), stride=2),
            DepthwiseSeparableConv(ch(128), ch(128), stride=1),
        )

        # Stage 2: stride 2 (output stride 8)
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(ch(128), ch(256), stride=2),
            DepthwiseSeparableConv(ch(256), ch(256), stride=1),
        )

        # Stage 3: stride 2 (output stride 16)
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(ch(256), ch(512), stride=2),
            DepthwiseSeparableConv(ch(512), ch(512), stride=1),
            DepthwiseSeparableConv(ch(512), ch(512), stride=1),
            DepthwiseSeparableConv(ch(512), ch(512), stride=1),
            DepthwiseSeparableConv(ch(512), ch(512), stride=1),
            DepthwiseSeparableConv(ch(512), ch(512), stride=1),
        )

        # Stage 4: stride 2 (output stride 32)
        self.stage4 = nn.Sequential(
            DepthwiseSeparableConv(ch(512), ch(1024), stride=2),
            DepthwiseSeparableConv(ch(1024), ch(1024), stride=1),
        )

        # Store output channels for each stage
        self.out_channels = [ch(32), ch(128), ch(256), ch(512), ch(1024)]

    def __call__(self, x: mx.array) -> List[mx.array]:
        """
        Forward pass returning features from specified stages.

        Args:
            x: Input tensor of shape (N, H, W, 3) in NHWC format.

        Returns:
            List of feature tensors from the specified return_stages.
        """
        features = []

        # Stage 0: Stem
        x = self.stem(x)
        if 0 in self.return_stages:
            features.append(x)

        # Stage 1
        x = self.stage1(x)
        if 1 in self.return_stages:
            features.append(x)

        # Stage 2
        x = self.stage2(x)
        if 2 in self.return_stages:
            features.append(x)

        # Stage 3
        x = self.stage3(x)
        if 3 in self.return_stages:
            features.append(x)

        # Stage 4
        x = self.stage4(x)
        if 4 in self.return_stages:
            features.append(x)

        return features

    def get_out_channels(self, stages: Optional[List[int]] = None) -> List[int]:
        """Get output channels for specified stages."""
        stages = stages or self.return_stages
        return [self.out_channels[s] for s in stages]


class MobileNetV2(nn.Module):
    """
    MobileNetV2 backbone for feature extraction.

    Uses inverted residual blocks with linear bottlenecks.
    Used as the backbone for RetinaFace (mnet_v2 variant) and recognition models.

    The architecture follows PyTorch's MobileNetV2 exactly:
    - features.0: Stem (Conv2d + BN + ReLU6)
    - features.1-17: Inverted residual blocks
    - features.18: Final 1x1 conv (320 -> 1280 channels)

    For RetinaFace, we extract features from:
    - features.6 (32ch, stride 8) -> return_features index 0
    - features.13 (96ch, stride 16) -> return_features index 1
    - features.18 (1280ch, stride 32) -> return_features index 2

    Args:
        width_mult: Width multiplier. Default: 1.0.
        return_features: Which feature indices to return. Default: [6, 13, 18].
            These correspond to PyTorch MobileNetV2 features.X indices.

    Returns:
        List of feature maps from specified feature indices.
    """

    def __init__(
        self,
        width_mult: float = 1.0,
        return_stages: Optional[List[int]] = None,
        return_features: Optional[List[int]] = None,
    ):
        super().__init__()
        self.width_mult = width_mult
        # Support both old API (return_stages) and new API (return_features)
        # Default to PyTorch feature indices for RetinaFace: [6, 13, 18]
        if return_features is not None:
            self.return_features = return_features
        elif return_stages is not None:
            # Map old stage indices to feature indices (approximate)
            stage_to_feature = {3: 6, 5: 13, 7: 18}
            self.return_features = [stage_to_feature.get(s, s) for s in return_stages]
        else:
            self.return_features = [6, 13, 18]

        def _make_divisible(v: float, divisor: int = 8) -> int:
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def ch(c: int) -> int:
            return _make_divisible(c * width_mult)

        # MobileNetV2 configuration: (expand_ratio, out_channels, num_blocks, stride)
        # These map to features.1 through features.17 in PyTorch
        # fmt: off
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],   # features.1 (stage 1)
            [6, 24, 2, 2],   # features.2-3 (stage 2)
            [6, 32, 3, 2],   # features.4-6 (stage 3) - stride 8
            [6, 64, 4, 2],   # features.7-10 (stage 4) - stride 16
            [6, 96, 3, 1],   # features.11-13 (stage 5) - stride 16
            [6, 160, 3, 2],  # features.14-16 (stage 6) - stride 32
            [6, 320, 1, 1],  # features.17 (stage 7) - stride 32
        ]
        # fmt: on

        # Stem (features.0) - MobileNetV2 uses ReLU6 throughout
        self.stem = ConvBNReLU6(3, ch(32), kernel_size=3, stride=2, padding=1)

        # Build inverted residual blocks as named attributes
        # Using setattr to properly register each stage as a submodule
        in_channels = ch(32)
        self._num_stages = len(inverted_residual_setting)

        # Track output channels for each feature index
        # features.0 = stem = 32ch
        self._feature_channels = {0: ch(32)}
        feature_idx = 1

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
                self._feature_channels[feature_idx] = out_channels
                feature_idx += 1
            # Set as named attribute so MLX registers parameters
            setattr(self, f'stage{stage_idx + 1}', nn.Sequential(*layers))

        # Final 1x1 conv (features.18): 320 -> 1280 with ReLU6
        self.final_conv = ConvBNReLU6(ch(320), ch(1280), kernel_size=1, stride=1, padding=0)
        self._feature_channels[18] = ch(1280)

        # For backward compatibility
        self.out_channels = [ch(32)]  # stem
        for t, c, n, s in inverted_residual_setting:
            self.out_channels.append(ch(c))
        self.out_channels.append(ch(1280))  # final conv

    def __call__(self, x: mx.array) -> List[mx.array]:
        """
        Forward pass returning features from specified feature indices.

        Args:
            x: Input tensor of shape (N, H, W, 3) in NHWC format.

        Returns:
            List of feature tensors from the specified return_features indices.
        """
        features = {}

        # Stem (features.0)
        x = self.stem(x)
        features[0] = x

        # Inverted residual blocks (features.1 through features.17)
        # Each stage contains multiple blocks, and we track each block's output
        feature_idx = 1

        # MobileNetV2 inverted_residual_setting for block counts
        block_counts = [1, 2, 3, 4, 3, 3, 1]  # n values from settings

        for stage_idx in range(self._num_stages):
            stage = getattr(self, f'stage{stage_idx + 1}')
            # Process each layer in the stage individually to get intermediate features
            for layer in stage.layers:
                x = layer(x)
                features[feature_idx] = x
                feature_idx += 1

        # Final 1x1 conv (features.18)
        x = self.final_conv(x)
        features[18] = x

        # Return only the requested features
        return [features[idx] for idx in self.return_features]

    def get_out_channels(self, stages: Optional[List[int]] = None) -> List[int]:
        """Get output channels for specified feature indices."""
        if stages is not None:
            # Map old stage indices to feature indices
            stage_to_feature = {3: 6, 5: 13, 7: 18}
            indices = [stage_to_feature.get(s, s) for s in stages]
        else:
            indices = self.return_features
        return [self._feature_channels[idx] for idx in indices]


class MobileNetV2Fused(nn.Module):
    """
    MobileNetV2 backbone with fused BatchNorm (for ONNX-converted weights).

    This variant is designed to load weights from ONNX models where BatchNorm
    has been fused into Conv layers during export optimization. The Conv layers
    have bias enabled since BN's parameters are folded into them.

    Architecture is identical to MobileNetV2, but uses InvertedResidualFused
    blocks instead of InvertedResidual, and ConvReLU6Fused instead of ConvBNReLU6.

    Args:
        width_mult: Width multiplier. Default: 1.0.
        return_features: Which feature indices to return. Default: [6, 13, 18].

    Returns:
        List of feature maps from specified feature indices.
    """

    def __init__(
        self,
        width_mult: float = 1.0,
        return_stages: Optional[List[int]] = None,
        return_features: Optional[List[int]] = None,
    ):
        super().__init__()
        self.width_mult = width_mult

        if return_features is not None:
            self.return_features = return_features
        elif return_stages is not None:
            stage_to_feature = {3: 6, 5: 13, 7: 18}
            self.return_features = [stage_to_feature.get(s, s) for s in return_stages]
        else:
            self.return_features = [6, 13, 18]

        def _make_divisible(v: float, divisor: int = 8) -> int:
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def ch(c: int) -> int:
            return _make_divisible(c * width_mult)

        # MobileNetV2 configuration: (expand_ratio, out_channels, num_blocks, stride)
        # fmt: off
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],   # features.1 (stage 1)
            [6, 24, 2, 2],   # features.2-3 (stage 2)
            [6, 32, 3, 2],   # features.4-6 (stage 3) - stride 8
            [6, 64, 4, 2],   # features.7-10 (stage 4) - stride 16
            [6, 96, 3, 1],   # features.11-13 (stage 5) - stride 16
            [6, 160, 3, 2],  # features.14-16 (stage 6) - stride 32
            [6, 320, 1, 1],  # features.17 (stage 7) - stride 32
        ]
        # fmt: on

        # Stem (features.0) - Fused Conv with ReLU6
        self.stem = ConvReLU6Fused(3, ch(32), kernel_size=3, stride=2, padding=1)

        # Build inverted residual blocks as named attributes
        in_channels = ch(32)
        self._num_stages = len(inverted_residual_setting)

        # Track output channels for each feature index
        self._feature_channels = {0: ch(32)}
        feature_idx = 1

        for stage_idx, (t, c, n, s) in enumerate(inverted_residual_setting):
            out_channels = ch(c)
            layers = []
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(
                    InvertedResidualFused(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride,
                        expand_ratio=t,
                    )
                )
                in_channels = out_channels
                self._feature_channels[feature_idx] = out_channels
                feature_idx += 1
            setattr(self, f'stage{stage_idx + 1}', nn.Sequential(*layers))

        # Final 1x1 conv (features.18): 320 -> 1280 with ReLU6
        self.final_conv = ConvReLU6Fused(ch(320), ch(1280), kernel_size=1, stride=1, padding=0)
        self._feature_channels[18] = ch(1280)

        # For backward compatibility
        self.out_channels = [ch(32)]
        for t, c, n, s in inverted_residual_setting:
            self.out_channels.append(ch(c))
        self.out_channels.append(ch(1280))

    def __call__(self, x: mx.array) -> List[mx.array]:
        """
        Forward pass returning features from specified feature indices.

        Args:
            x: Input tensor of shape (N, H, W, 3) in NHWC format.

        Returns:
            List of feature tensors from the specified return_features indices.
        """
        features = {}

        # Stem (features.0)
        x = self.stem(x)
        features[0] = x

        # Inverted residual blocks (features.1 through features.17)
        feature_idx = 1

        for stage_idx in range(self._num_stages):
            stage = getattr(self, f'stage{stage_idx + 1}')
            for layer in stage.layers:
                x = layer(x)
                features[feature_idx] = x
                feature_idx += 1

        # Final 1x1 conv (features.18)
        x = self.final_conv(x)
        features[18] = x

        return [features[idx] for idx in self.return_features]

    def get_out_channels(self, stages: Optional[List[int]] = None) -> List[int]:
        """Get output channels for specified feature indices."""
        if stages is not None:
            stage_to_feature = {3: 6, 5: 13, 7: 18}
            indices = [stage_to_feature.get(s, s) for s in stages]
        else:
            indices = self.return_features
        return [self._feature_channels[idx] for idx in indices]
