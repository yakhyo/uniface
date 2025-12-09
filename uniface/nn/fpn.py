# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo
#
# MLX Feature Pyramid Network and Context Modules

"""
Feature Pyramid Network (FPN) and SSH context module for face detection.

These modules are used in RetinaFace and SCRFD for multi-scale feature fusion.
The implementations match the original PyTorch RetinaFace architecture for
weight compatibility.
"""

from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn

from uniface.nn.conv import ConvBN, ConvBNReLU

__all__ = [
    'FPN',
    'FPNFused',
    'SSH',
    'SSHFused',
]


class _ConvBN(nn.Module):
    """
    Conv + BatchNorm layer matching PyTorch RetinaFace structure.

    PyTorch stores as sequential with indices:
    - 0: Conv2d
    - 1: BatchNorm2d

    We match this by storing as 'conv' and 'bn' attributes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        return self.bn(self.conv(x))


class FPN(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature fusion.

    This implementation matches the PyTorch RetinaFace FPN structure:
    - output1, output2, output3: 1x1 lateral convolutions (one per level)
    - merge1, merge2: 3x3 smoothing convolutions after top-down fusion

    Weight key structure:
    - fpn.output1.conv.weight, fpn.output1.bn.weight, etc.
    - fpn.merge1.conv.weight, fpn.merge1.bn.weight, etc.

    Args:
        in_channels_list: List of input channel counts from backbone stages.
        out_channels: Number of output channels for all FPN levels.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
    ):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        assert len(in_channels_list) == 3, 'FPN expects exactly 3 input levels'

        # Lateral connections (1x1 convs) - named to match PyTorch
        # output1 corresponds to the smallest feature (stride 8)
        # output3 corresponds to the largest feature (stride 32)
        self.output1 = _ConvBN(in_channels_list[0], out_channels, kernel_size=1, padding=0)
        self.output2 = _ConvBN(in_channels_list[1], out_channels, kernel_size=1, padding=0)
        self.output3 = _ConvBN(in_channels_list[2], out_channels, kernel_size=1, padding=0)

        # Smoothing convolutions (3x3) after top-down fusion
        self.merge1 = _ConvBN(out_channels, out_channels, kernel_size=3, padding=1)
        self.merge2 = _ConvBN(out_channels, out_channels, kernel_size=3, padding=1)

    def __call__(self, features: List[mx.array]) -> List[mx.array]:
        """
        Forward pass for FPN.

        Args:
            features: List of feature maps from backbone, ordered from
                      high resolution (stride 8) to low resolution (stride 32).
                      For MobileNetV2: [stage3_output, stage5_output, stage7_output]

        Returns:
            List of FPN feature maps with consistent channel dimensions.
        """
        assert len(features) == 3, f'Expected 3 features, got {len(features)}'

        c1, c2, c3 = features  # c1 is stride 8, c3 is stride 32

        # Lateral connections (1x1 conv)
        p3 = self.output3(c3)  # Smallest spatial, largest stride
        p2 = self.output2(c2)
        p1 = self.output1(c1)  # Largest spatial, smallest stride

        # Top-down pathway: upsample and add
        # p2 = p2 + upsample(p3)
        h2, w2 = p2.shape[1:3]
        p2 = p2 + self._upsample(p3, (h2, w2))

        # p1 = p1 + upsample(p2)
        h1, w1 = p1.shape[1:3]
        p1 = p1 + self._upsample(p2, (h1, w1))

        # Smoothing convolutions (3x3)
        p1 = self.merge1(p1)
        p2 = self.merge2(p2)
        # Note: p3 doesn't get merged in the original RetinaFace

        return [p1, p2, p3]

    def _upsample(self, x: mx.array, target_size: Tuple[int, int]) -> mx.array:
        """
        Upsample feature map using nearest neighbor interpolation.

        Args:
            x: Input tensor of shape (N, H, W, C) in NHWC format.
            target_size: Target (height, width).

        Returns:
            Upsampled tensor.
        """
        n, h, w, c = x.shape
        target_h, target_w = target_size

        scale_h = target_h // h
        scale_w = target_w // w

        # Use repeat for nearest neighbor upsampling
        # Reshape to (N, H, 1, W, 1, C) then tile
        x = mx.expand_dims(x, axis=2)  # (N, H, 1, W, C)
        x = mx.expand_dims(x, axis=4)  # (N, H, 1, W, 1, C)

        # Tile to get (N, H, scale_h, W, scale_w, C)
        x = mx.tile(x, (1, 1, scale_h, 1, scale_w, 1))

        # Reshape to (N, H*scale_h, W*scale_w, C)
        x = mx.reshape(x, (n, h * scale_h, w * scale_w, c))

        return x


class SSH(nn.Module):
    """
    Single Stage Headless (SSH) context module.

    This implementation matches the PyTorch RetinaFace SSH structure exactly:
    - conv3X3: 3x3 convolution (half of output channels)
    - conv5X5_1 + conv5X5_2: Two 3x3 convs for 5x5 effective receptive field
    - conv7X7_2 + conv7x7_3: Two 3x3 convs for 7x7 effective receptive field

    Note: The original PyTorch model has inconsistent naming (conv7X7_2 vs conv7x7_3).
    We match this exactly for weight compatibility.

    Weight key structure:
    - ssh1.conv3X3.conv.weight, ssh1.conv3X3.bn.weight, etc.
    - ssh1.conv5X5_1.conv.weight, ssh1.conv5X5_2.conv.weight, etc.
    - ssh1.conv7X7_2.conv.weight, ssh1.conv7x7_3.conv.weight, etc.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        # Calculate branch channel sizes
        # PyTorch structure uses: conv3X3 = out_channels // 2
        # conv5X5 and conv7X7 each contribute out_channels // 4
        half_channels = out_channels // 2
        quarter_channels = out_channels // 4

        # Branch 1: 3x3 conv (no activation after BN)
        self.conv3X3 = _ConvBN(in_channels, half_channels, kernel_size=3, padding=1)

        # Branch 2: 5x5 effective receptive field
        # First conv with ReLU, second without
        self.conv5X5_1 = _ConvBN(in_channels, quarter_channels, kernel_size=3, padding=1)
        self.conv5X5_2 = _ConvBN(quarter_channels, quarter_channels, kernel_size=3, padding=1)

        # Branch 3: 7x7 effective receptive field
        # Note: Uses conv5X5_1 output as input (shared first layer)
        # conv7X7_2 and conv7x7_3 (note the case difference in original PyTorch)
        self.conv7X7_2 = _ConvBN(quarter_channels, quarter_channels, kernel_size=3, padding=1)
        self.conv7x7_3 = _ConvBN(quarter_channels, quarter_channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass for SSH module.

        Args:
            x: Input feature map of shape (N, H, W, C).

        Returns:
            Enhanced feature map with context information.
        """
        # Branch 1: 3x3 conv
        out_3x3 = self.conv3X3(x)

        # Branch 2: 5x5 effective RF
        out_5x5_1 = nn.relu(self.conv5X5_1(x))
        out_5x5 = self.conv5X5_2(out_5x5_1)

        # Branch 3: 7x7 effective RF (builds on 5x5 first layer)
        out_7x7_2 = nn.relu(self.conv7X7_2(out_5x5_1))
        out_7x7 = self.conv7x7_3(out_7x7_2)

        # Concatenate all branches along channel dimension (last dim in NHWC)
        out = mx.concatenate([out_3x3, out_5x5, out_7x7], axis=-1)

        # Apply ReLU after concatenation
        out = nn.relu(out)

        return out


class _ConvFused(nn.Module):
    """
    Fused Conv layer (BatchNorm folded into Conv).

    Used for loading ONNX weights where BatchNorm has been fused.
    The Conv layer has bias enabled.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,  # Fused weights include bias
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class _ConvLeakyReLUFused(nn.Module):
    """
    Fused Conv + LeakyReLU layer (BatchNorm folded into Conv).

    Used for FPN where LeakyReLU is applied after Conv+BN.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        negative_slope: float = 0.0,  # ONNX uses 0.0 (which is ReLU, but modeled as LeakyReLU)
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.negative_slope = negative_slope

    def __call__(self, x: mx.array) -> mx.array:
        out = self.conv(x)
        # LeakyReLU with slope 0.0 is equivalent to ReLU
        return nn.leaky_relu(out, negative_slope=self.negative_slope)


class FPNFused(nn.Module):
    """
    Feature Pyramid Network with fused BatchNorm (for ONNX-converted weights).

    This variant uses _ConvLeakyReLUFused for loading ONNX models where
    BatchNorm has been fused into Conv layers and LeakyReLU follows.

    ONNX FPN structure:
    - output1/2/3: 1x1 Conv + BN + LeakyReLU (lateral connections)
    - merge1/2: 3x3 Conv + BN + LeakyReLU (smoothing)

    Output order matches ONNX: [P1 (stride 8), P2 (stride 16), P3 (stride 32)]

    Args:
        in_channels_list: List of input channel counts from backbone stages.
        out_channels: Number of output channels for all FPN levels.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
    ):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        assert len(in_channels_list) == 3, 'FPN expects exactly 3 input levels'

        # Lateral connections (1x1 convs) with LeakyReLU
        self.output1 = _ConvLeakyReLUFused(in_channels_list[0], out_channels, kernel_size=1, padding=0)
        self.output2 = _ConvLeakyReLUFused(in_channels_list[1], out_channels, kernel_size=1, padding=0)
        self.output3 = _ConvLeakyReLUFused(in_channels_list[2], out_channels, kernel_size=1, padding=0)

        # Smoothing convolutions (3x3) with LeakyReLU
        self.merge1 = _ConvLeakyReLUFused(out_channels, out_channels, kernel_size=3, padding=1)
        self.merge2 = _ConvLeakyReLUFused(out_channels, out_channels, kernel_size=3, padding=1)

    def __call__(self, features: List[mx.array]) -> List[mx.array]:
        """
        Forward pass for FPN.

        IMPORTANT: The ONNX FPN applies merge BEFORE upsampling to the next level.
        Order of operations (matching ONNX exactly):
        1. Lateral connections: p1, p2, p3 from backbone features
        2. P3 → upsample → add to P2 → merge2 → P2 output
        3. P2 (merged) → upsample → add to P1 → merge1 → P1 output
        """
        assert len(features) == 3, f'Expected 3 features, got {len(features)}'

        c1, c2, c3 = features  # c1 is stride 8 (80x80), c3 is stride 32 (20x20)

        # Lateral connections with LeakyReLU
        p3 = self.output3(c3)  # Stride 32 (20x20)
        p2_lat = self.output2(c2)  # Stride 16 (40x40)
        p1_lat = self.output1(c1)  # Stride 8 (80x80)

        # Top-down pathway (ONNX order):
        # Step 1: P2 = P2_lateral + upsample(P3), then merge
        h2, w2 = p2_lat.shape[1:3]
        p2 = p2_lat + self._upsample(p3, (h2, w2))
        p2 = self.merge2(p2)  # Merge BEFORE upsampling to P1

        # Step 2: P1 = P1_lateral + upsample(P2_merged), then merge
        h1, w1 = p1_lat.shape[1:3]
        p1 = p1_lat + self._upsample(p2, (h1, w1))  # Uses MERGED p2
        p1 = self.merge1(p1)

        # Return in order: [P1 (stride 8), P2 (stride 16), P3 (stride 32)]
        return [p1, p2, p3]

    def _upsample(self, x: mx.array, target_size: Tuple[int, int]) -> mx.array:
        """Upsample using nearest neighbor interpolation."""
        n, h, w, c = x.shape
        target_h, target_w = target_size

        scale_h = target_h // h
        scale_w = target_w // w

        x = mx.expand_dims(x, axis=2)
        x = mx.expand_dims(x, axis=4)
        x = mx.tile(x, (1, 1, scale_h, 1, scale_w, 1))
        x = mx.reshape(x, (n, h * scale_h, w * scale_w, c))

        return x


class SSHFused(nn.Module):
    """
    SSH context module with fused BatchNorm (for ONNX-converted weights).

    This variant uses _ConvFused instead of _ConvBN for loading ONNX models
    where BatchNorm has been fused into Conv layers.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        half_channels = out_channels // 2
        quarter_channels = out_channels // 4

        # Branch 1: 3x3 conv
        self.conv3X3 = _ConvFused(in_channels, half_channels, kernel_size=3, padding=1)

        # Branch 2: 5x5 effective receptive field
        self.conv5X5_1 = _ConvFused(in_channels, quarter_channels, kernel_size=3, padding=1)
        self.conv5X5_2 = _ConvFused(quarter_channels, quarter_channels, kernel_size=3, padding=1)

        # Branch 3: 7x7 effective receptive field
        self.conv7X7_2 = _ConvFused(quarter_channels, quarter_channels, kernel_size=3, padding=1)
        self.conv7x7_3 = _ConvFused(quarter_channels, quarter_channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass for SSH module."""
        # Branch 1
        out_3x3 = self.conv3X3(x)

        # Branch 2
        out_5x5_1 = nn.relu(self.conv5X5_1(x))
        out_5x5 = self.conv5X5_2(out_5x5_1)

        # Branch 3
        out_7x7_2 = nn.relu(self.conv7X7_2(out_5x5_1))
        out_7x7 = self.conv7x7_3(out_7x7_2)

        # Concatenate and activate
        out = mx.concatenate([out_3x3, out_5x5, out_7x7], axis=-1)
        out = nn.relu(out)

        return out
