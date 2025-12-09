# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo
#
# MLX Convolution Building Blocks

"""
Convolution building blocks for MLX neural networks.

This module provides common convolution patterns used in face detection
and recognition models, implemented in MLX for Apple Silicon.

Note: MLX uses NHWC format (channels-last), unlike PyTorch's NCHW.
"""

from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    'ConvBN',
    'ConvBNReLU',
    'ConvBNReLU6',
    'ConvBNSiLU',
    'ConvReLU6Fused',
    'DepthwiseSeparableConv',
    'InvertedResidual',
    'InvertedResidualFused',
    'Bottleneck',
    'C3',
    'SPPF',
    'Concat',
]


class ConvBN(nn.Module):
    """
    Convolution + BatchNorm block.

    A basic building block combining a 2D convolution with batch normalization.
    No activation is applied.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel.
        stride: Stride of the convolution. Default: 1.
        padding: Padding added to input. Default: 0.
        groups: Number of blocked connections. Default: 1.
        bias: If True, adds a learnable bias. Default: False (BN handles bias).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass."""
        return self.bn(self.conv(x))


class ConvBNReLU(nn.Module):
    """
    Convolution + BatchNorm + ReLU block.

    A common building block used extensively in MobileNet and detection networks.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel.
        stride: Stride of the convolution. Default: 1.
        padding: Padding added to input. Default: 0.
        groups: Number of blocked connections. Default: 1.
        bias: If True, adds a learnable bias. Default: False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with ReLU activation."""
        return nn.relu(self.bn(self.conv(x)))


class ConvBNReLU6(nn.Module):
    """
    Convolution + BatchNorm + ReLU6 block.

    Used in MobileNetV2 for bounded activations.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel.
        stride: Stride of the convolution. Default: 1.
        padding: Padding added to input. Default: 0.
        groups: Number of blocked connections. Default: 1.
        bias: If True, adds a learnable bias. Default: False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with ReLU6 activation (clamped to [0, 6])."""
        return mx.clip(nn.relu(self.bn(self.conv(x))), 0, 6)


class ConvReLU6Fused(nn.Module):
    """
    Fused Convolution + ReLU6 block (BatchNorm folded into Conv).

    This is used when loading weights from ONNX models where BatchNorm
    has been fused into the Conv layer during export optimization.
    The Conv layer has bias enabled since BN's bias is folded into it.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel.
        stride: Stride of the convolution. Default: 1.
        padding: Padding added to input. Default: 0.
        groups: Number of blocked connections. Default: 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=True,  # Fused weights include bias from BatchNorm
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with ReLU6 activation (clamped to [0, 6])."""
        return mx.clip(nn.relu(self.conv(x)), 0, 6)


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution block.

    This is the core building block of MobileNetV1, consisting of:
    1. Depthwise convolution (spatial filtering per channel)
    2. Pointwise convolution (1x1 conv for channel mixing)

    Both followed by BatchNorm and ReLU.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride of the depthwise convolution. Default: 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()
        # Depthwise convolution (groups=in_channels for per-channel filtering)
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,  # Critical: enables true depthwise convolution
            bias=False,
        )
        self.bn1 = nn.BatchNorm(in_channels)

        # Pointwise convolution (1x1 conv for channel mixing)
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn2 = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass."""
        # Depthwise
        x = nn.relu(self.bn1(self.depthwise(x)))
        # Pointwise
        x = nn.relu(self.bn2(self.pointwise(x)))
        return x


class InvertedResidual(nn.Module):
    """
    Inverted Residual block (MobileNetV2 building block).

    This block uses:
    1. Expansion: 1x1 conv to expand channels
    2. Depthwise: 3x3 depthwise conv
    3. Projection: 1x1 conv to reduce channels

    With residual connection when stride=1 and in_channels=out_channels.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride of the depthwise convolution. Default: 1.
        expand_ratio: Expansion factor for intermediate channels. Default: 6.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 6,
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels

        hidden_dim = in_channels * expand_ratio

        layers = []

        # Expansion (only if expand_ratio > 1)
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm(hidden_dim))
            # ReLU6 will be applied in forward

        self.expand = nn.Sequential(*layers) if layers else None
        self.expand_dim = hidden_dim if expand_ratio != 1 else in_channels

        # Depthwise convolution (groups=channels for per-channel filtering)
        self.depthwise = nn.Conv2d(
            self.expand_dim,
            self.expand_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=self.expand_dim,  # Critical: enables true depthwise convolution
            bias=False,
        )
        self.bn_dw = nn.BatchNorm(self.expand_dim)

        # Projection (linear, no activation)
        self.project = nn.Conv2d(
            self.expand_dim,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn_proj = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with optional residual connection."""
        identity = x

        # Expansion
        if self.expand is not None:
            out = self.expand(x)
            out = mx.clip(nn.relu(out), 0, 6)  # ReLU6
        else:
            out = x

        # Depthwise
        out = self.depthwise(out)
        out = mx.clip(nn.relu(self.bn_dw(out)), 0, 6)  # ReLU6

        # Projection (linear)
        out = self.bn_proj(self.project(out))

        # Residual connection
        if self.use_residual:
            out = out + identity

        return out


class InvertedResidualFused(nn.Module):
    """
    Fused Inverted Residual block (BatchNorm folded into Conv).

    This is the MobileNetV2 building block for loading ONNX models where
    BatchNorm has been fused into Conv layers during export optimization.

    Structure:
    1. Expansion: 1x1 conv (if expand_ratio > 1) with ReLU6
    2. Depthwise: 3x3 depthwise conv with ReLU6
    3. Projection: 1x1 conv (linear, no activation)

    With residual connection when stride=1 and in_channels=out_channels.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride of the depthwise convolution. Default: 1.
        expand_ratio: Expansion factor for intermediate channels. Default: 6.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 6,
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels

        hidden_dim = in_channels * expand_ratio

        # Expansion (only if expand_ratio > 1)
        # Using nn.Sequential to match the unfused version's structure for weight loading
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=True),
            )
        else:
            self.expand = None
        self.expand_dim = hidden_dim if expand_ratio != 1 else in_channels

        # Depthwise convolution (groups=channels for per-channel filtering)
        self.depthwise = nn.Conv2d(
            self.expand_dim,
            self.expand_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=self.expand_dim,
            bias=True,  # Fused weights include bias
        )

        # Projection (linear, no activation)
        self.project = nn.Conv2d(
            self.expand_dim,
            out_channels,
            kernel_size=1,
            bias=True,  # Fused weights include bias
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with optional residual connection."""
        identity = x

        # Expansion
        if self.expand is not None:
            out = self.expand(x)
            out = mx.clip(nn.relu(out), 0, 6)  # ReLU6
        else:
            out = x

        # Depthwise with ReLU6
        out = mx.clip(nn.relu(self.depthwise(out)), 0, 6)

        # Projection (linear, no activation)
        out = self.project(out)

        # Residual connection
        if self.use_residual:
            out = out + identity

        return out


class ConvBNSiLU(nn.Module):
    """
    Convolution + BatchNorm + SiLU (Swish) block.

    This is the standard convolution block used in YOLOv5.
    SiLU (Sigmoid Linear Unit) = x * sigmoid(x), also known as Swish.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel. Default: 1.
        stride: Stride of the convolution. Default: 1.
        padding: Padding added to input. If None, uses autopad.
        groups: Number of blocked connections. Default: 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
    ):
        super().__init__()
        # Auto-pad to maintain spatial dimensions for stride=1
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with SiLU activation."""
        return nn.silu(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """
    Standard bottleneck block used in YOLOv5 C3 modules.

    Structure: Conv(1x1) -> Conv(3x3) with optional shortcut.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        shortcut: Whether to add residual connection. Default: True.
        groups: Number of groups for grouped convolution. Default: 1.
        expansion: Hidden channels = out_channels * expansion. Default: 0.5.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        groups: int = 1,
        expansion: float = 0.5,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = ConvBNSiLU(in_channels, hidden_channels, kernel_size=1)
        self.cv2 = ConvBNSiLU(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.add = shortcut and in_channels == out_channels

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with optional shortcut."""
        out = self.cv2(self.cv1(x))
        if self.add:
            out = out + x
        return out


class C3(nn.Module):
    """
    CSP Bottleneck with 3 convolutions.

    This is the main building block of YOLOv5's backbone and neck.
    It splits the input, processes one path through bottleneck blocks,
    and concatenates with the other path.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        n: Number of bottleneck blocks. Default: 1.
        shortcut: Whether to use shortcut in bottlenecks. Default: True.
        groups: Number of groups. Default: 1.
        expansion: Hidden channels expansion factor. Default: 0.5.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        shortcut: bool = True,
        groups: int = 1,
        expansion: float = 0.5,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.cv1 = ConvBNSiLU(in_channels, hidden_channels, kernel_size=1)
        self.cv2 = ConvBNSiLU(in_channels, hidden_channels, kernel_size=1)
        self.cv3 = ConvBNSiLU(2 * hidden_channels, out_channels, kernel_size=1)

        # Sequential bottleneck blocks
        self.m = nn.Sequential(
            *[Bottleneck(hidden_channels, hidden_channels, shortcut, groups, expansion=1.0) for _ in range(n)]
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with CSP structure."""
        # Split path: cv1 -> bottlenecks, cv2 -> direct
        path1 = self.m(self.cv1(x))
        path2 = self.cv2(x)
        # Concatenate along channel dimension (last dim in NHWC)
        return self.cv3(mx.concatenate([path1, path2], axis=-1))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF).

    This module applies multiple max pooling operations at different scales
    and concatenates the results. It's faster than the original SPP by
    using sequential pooling instead of parallel.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size for max pooling. Default: 5.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = ConvBNSiLU(in_channels, hidden_channels, kernel_size=1)
        self.cv2 = ConvBNSiLU(hidden_channels * 4, out_channels, kernel_size=1)
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with spatial pyramid pooling."""
        x = self.cv1(x)

        # Sequential max pooling (SPPF is faster than parallel SPP)
        # In NHWC format, we pool over H and W (axes 1, 2)
        y1 = mx.max(
            mx.pad(x, [(0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)]),
            axis=(1, 2),
            keepdims=True,
        )
        # Expand y1 back to spatial size of x for proper pooling
        # Actually, we need to use sliding window max pool
        # MLX doesn't have max_pool2d built-in, so we'll implement it manually
        y1 = self._max_pool2d(x, self.kernel_size, stride=1)
        y2 = self._max_pool2d(y1, self.kernel_size, stride=1)
        y3 = self._max_pool2d(y2, self.kernel_size, stride=1)

        return self.cv2(mx.concatenate([x, y1, y2, y3], axis=-1))

    def _max_pool2d(self, x: mx.array, kernel_size: int, stride: int = 1) -> mx.array:
        """Apply 2D max pooling in NHWC format."""
        padding = kernel_size // 2
        # Pad spatial dimensions (H, W are axes 1 and 2 in NHWC)
        x_padded = mx.pad(x, [(0, 0), (padding, padding), (padding, padding), (0, 0)])
        n, h, w, c = x_padded.shape

        # Output size
        out_h = (h - kernel_size) // stride + 1
        out_w = (w - kernel_size) // stride + 1

        # Create windows using as_strided-like approach
        # For simplicity with stride=1, we can use a loop (this is inefficient but correct)
        # A more efficient implementation would use im2col or native ops when available

        # For stride=1 pooling, we can accumulate max over shifted versions
        result = x_padded[:, :out_h, :out_w, :]
        for i in range(kernel_size):
            for j in range(kernel_size):
                if i == 0 and j == 0:
                    continue
                result = mx.maximum(result, x_padded[:, i : i + out_h, j : j + out_w, :])

        return result


class Concat(nn.Module):
    """
    Concatenate tensors along a specified dimension.

    Used in YOLOv5's PANet neck for feature fusion.

    Args:
        dimension: Dimension to concatenate along. Default: -1 (channels in NHWC).
    """

    def __init__(self, dimension: int = -1):
        super().__init__()
        self.d = dimension

    def __call__(self, x: list) -> mx.array:
        """Concatenate list of tensors."""
        return mx.concatenate(x, axis=self.d)
