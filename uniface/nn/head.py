# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo
#
# MLX Detection Heads

"""
Detection heads for face detection models.

These heads take FPN features and predict:
- Classification scores (face vs background)
- Bounding box coordinates
- Facial landmarks

The structure matches PyTorch RetinaFace for weight compatibility.
"""

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    'ClassHead',
    'BboxHead',
    'LandmarkHead',
    'ClassHeadWrapper',
    'BboxHeadWrapper',
    'LandmarkHeadWrapper',
]


class _HeadConv(nn.Module):
    """
    Single convolution for detection head.

    PyTorch uses a list of convolutions stored as class_head.0, class_head.1, etc.
    We use setattr to create attributes with numeric names for weight compatibility.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = mx.zeros((out_channels, 1, 1, in_channels))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # Manual conv2d with 1x1 kernel
        return mx.conv2d(x, self.weight, padding=0) + self.bias


class ClassHeadWrapper(nn.Module):
    """
    Classification head wrapper matching PyTorch structure.

    PyTorch structure:
    - class_head.class_head.0.weight/bias (for FPN level 0)
    - class_head.class_head.1.weight/bias (for FPN level 1)
    - class_head.class_head.2.weight/bias (for FPN level 2)

    This wrapper creates the nested structure for weight loading.
    """

    def __init__(self, in_channels: int, num_anchors: int = 2, num_classes: int = 2):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        out_channels = num_anchors * num_classes

        # Create nested structure matching PyTorch
        # class_head contains a list-like structure with indices 0, 1, 2
        self.class_head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward_level(self, x: mx.array, level: int) -> mx.array:
        """Forward pass for a specific FPN level."""
        conv = self.class_head.layers[level]
        out = conv(x)

        # Reshape to (N, H*W*num_anchors, num_classes)
        n, h, w, _ = out.shape
        out = mx.reshape(out, (n, h * w * self.num_anchors, self.num_classes))
        return out


class BboxHeadWrapper(nn.Module):
    """
    Bounding box head wrapper matching PyTorch structure.

    PyTorch structure:
    - bbox_head.bbox_head.0.weight/bias
    - bbox_head.bbox_head.1.weight/bias
    - bbox_head.bbox_head.2.weight/bias
    """

    def __init__(self, in_channels: int, num_anchors: int = 2):
        super().__init__()
        self.num_anchors = num_anchors
        out_channels = num_anchors * 4  # 4 values per anchor

        self.bbox_head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward_level(self, x: mx.array, level: int) -> mx.array:
        """Forward pass for a specific FPN level."""
        conv = self.bbox_head.layers[level]
        out = conv(x)

        # Reshape to (N, H*W*num_anchors, 4)
        n, h, w, _ = out.shape
        out = mx.reshape(out, (n, h * w * self.num_anchors, 4))
        return out


class LandmarkHeadWrapper(nn.Module):
    """
    Landmark head wrapper matching PyTorch structure.

    PyTorch structure:
    - landmark_head.landmark_head.0.weight/bias
    - landmark_head.landmark_head.1.weight/bias
    - landmark_head.landmark_head.2.weight/bias
    """

    def __init__(self, in_channels: int, num_anchors: int = 2, num_landmarks: int = 5):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_landmarks = num_landmarks
        out_channels = num_anchors * num_landmarks * 2  # 2 coords per landmark

        self.landmark_head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward_level(self, x: mx.array, level: int) -> mx.array:
        """Forward pass for a specific FPN level."""
        conv = self.landmark_head.layers[level]
        out = conv(x)

        # Reshape to (N, H*W*num_anchors, num_landmarks * 2)
        n, h, w, _ = out.shape
        out = mx.reshape(out, (n, h * w * self.num_anchors, self.num_landmarks * 2))
        return out


# Legacy single-level heads for compatibility
class ClassHead(nn.Module):
    """
    Classification head for face detection (single level).

    Predicts face/background scores for each anchor.

    Args:
        in_channels: Number of input channels from FPN.
        num_anchors: Number of anchors per spatial location. Default: 2.
        num_classes: Number of classes (usually 2: face/background). Default: 2.
    """

    def __init__(
        self,
        in_channels: int,
        num_anchors: int = 2,
        num_classes: int = 2,
    ):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # Simple 1x1 conv for classification
        self.conv = nn.Conv2d(
            in_channels,
            num_anchors * num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass for classification head."""
        out = self.conv(x)
        n, h, w, _ = out.shape
        out = mx.reshape(out, (n, h * w * self.num_anchors, self.num_classes))
        return out


class BboxHead(nn.Module):
    """
    Bounding box regression head for face detection (single level).

    Predicts bounding box deltas (dx, dy, dw, dh) for each anchor.

    Args:
        in_channels: Number of input channels from FPN.
        num_anchors: Number of anchors per spatial location. Default: 2.
    """

    def __init__(
        self,
        in_channels: int,
        num_anchors: int = 2,
    ):
        super().__init__()
        self.num_anchors = num_anchors

        # 4 values per anchor: dx, dy, dw, dh
        self.conv = nn.Conv2d(
            in_channels,
            num_anchors * 4,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass for bounding box head."""
        out = self.conv(x)
        n, h, w, _ = out.shape
        out = mx.reshape(out, (n, h * w * self.num_anchors, 4))
        return out


class LandmarkHead(nn.Module):
    """
    Facial landmark regression head for face detection (single level).

    Predicts 5-point facial landmarks (eyes, nose, mouth corners).

    Args:
        in_channels: Number of input channels from FPN.
        num_anchors: Number of anchors per spatial location. Default: 2.
        num_landmarks: Number of landmark points. Default: 5.
    """

    def __init__(
        self,
        in_channels: int,
        num_anchors: int = 2,
        num_landmarks: int = 5,
    ):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_landmarks = num_landmarks

        # 2 values (x, y) per landmark, per anchor
        self.conv = nn.Conv2d(
            in_channels,
            num_anchors * num_landmarks * 2,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass for landmark head."""
        out = self.conv(x)
        n, h, w, _ = out.shape
        out = mx.reshape(out, (n, h * w * self.num_anchors, self.num_landmarks * 2))
        return out
