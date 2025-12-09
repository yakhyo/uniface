# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo
#
# MLX Neural Network Building Blocks for UniFace

"""
MLX neural network building blocks for UniFace.

This package provides reusable MLX neural network components for building
face detection, recognition, and analysis models optimized for Apple Silicon.

Modules:
    conv: Convolution layers (ConvBNReLU, DepthwiseSeparable, etc.)
    backbone: Backbone networks (MobileNetV1, MobileNetV2, ResNet)
    fpn: Feature Pyramid Network and SSH context module
    head: Detection heads (ClassHead, BboxHead, LandmarkHead)
"""

from uniface.mlx_utils import is_mlx_available

# Only import MLX modules if MLX is available
if is_mlx_available():
    from uniface.nn.backbone import (
        MobileNetV1,
        MobileNetV2,
    )
    from uniface.nn.conv import (
        C3,
        SPPF,
        Bottleneck,
        Concat,
        ConvBN,
        ConvBNReLU,
        ConvBNReLU6,
        ConvBNSiLU,
        DepthwiseSeparableConv,
        InvertedResidual,
    )
    from uniface.nn.fpn import (
        FPN,
        SSH,
    )
    from uniface.nn.head import (
        BboxHead,
        ClassHead,
        LandmarkHead,
    )

    __all__ = [
        # Convolution modules
        'ConvBN',
        'ConvBNReLU',
        'ConvBNReLU6',
        'ConvBNSiLU',
        'DepthwiseSeparableConv',
        'InvertedResidual',
        # YOLOv5 modules
        'Bottleneck',
        'C3',
        'SPPF',
        'Concat',
        # Backbone networks
        'MobileNetV1',
        'MobileNetV2',
        # FPN modules
        'FPN',
        'SSH',
        # Detection heads
        'ClassHead',
        'BboxHead',
        'LandmarkHead',
    ]
else:
    __all__ = []
