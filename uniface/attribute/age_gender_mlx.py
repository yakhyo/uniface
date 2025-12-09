# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo
#
# MLX Age and Gender Prediction Implementation

"""
Age and gender prediction model implemented in MLX for Apple Silicon.

This module provides an MLX implementation of the age/gender attribute
predictor that predicts:
- Gender: Female (0) or Male (1)
- Age: Estimated age in years
"""

from typing import List, Optional, Tuple, Union

import cv2
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from uniface.attribute.base import Attribute
from uniface.constants import AgeGenderWeights
from uniface.face_utils import bbox_center_alignment
from uniface.log import Logger
from uniface.mlx_utils import load_mlx_weights, synchronize, to_numpy
from uniface.model_store import get_weights_path
from uniface.nn.conv import ConvBNReLU, DepthwiseSeparableConv

__all__ = ['AgeGenderMLX']


class AgeGenderBackbone(nn.Module):
    """
    Lightweight CNN backbone for age and gender prediction.

    Uses a MobileNet-style architecture for efficient inference.

    Architecture:
    - Stem + 4 downsampling stages
    - Global average pooling
    - FC layer for age (1) and gender (2) outputs

    Output: 3 values [gender_logit_0, gender_logit_1, normalized_age]
    """

    def __init__(self, input_size: Tuple[int, int] = (96, 96), width_mult: float = 1.0):
        super().__init__()

        def ch(c: int) -> int:
            return max(int(c * width_mult), 8)

        # Stem
        self.stem = ConvBNReLU(3, ch(32), kernel_size=3, stride=2, padding=1)

        # Stages
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(ch(32), ch(64), stride=1),
            DepthwiseSeparableConv(ch(64), ch(64), stride=2),
        )

        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(ch(64), ch(128), stride=1),
            DepthwiseSeparableConv(ch(128), ch(128), stride=2),
        )

        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(ch(128), ch(256), stride=1),
            DepthwiseSeparableConv(ch(256), ch(256), stride=2),
        )

        self.stage4 = nn.Sequential(
            DepthwiseSeparableConv(ch(256), ch(512), stride=1),
            DepthwiseSeparableConv(ch(512), ch(512), stride=2),
        )

        # Final FC layer
        # For 96x96 input: 96 -> 48 -> 24 -> 12 -> 6 -> 3
        # After global avg pool: (N, C)
        self.fc = nn.Linear(ch(512), 3)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor (N, H, W, 3) in NHWC format.

        Returns:
            Predictions of shape (N, 3) = [gender_0, gender_1, age].
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Global average pooling
        x = mx.mean(x, axis=(1, 2))  # (N, C)

        # Predict
        x = self.fc(x)

        return x


class AgeGenderMLX(Attribute):
    """
    Age and gender prediction model using MLX backend.

    Predicts:
    - Gender ID: 0 (Female) or 1 (Male)
    - Age: Estimated age in years

    Args:
        model_name: Model weights to use.
        input_size: Input image size. If None, auto-detected.
    """

    def __init__(
        self,
        model_name: AgeGenderWeights = AgeGenderWeights.DEFAULT,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        Logger.info(f'Initializing AgeGender (MLX) with model={model_name.name}')

        self.input_size = input_size or (96, 96)

        # Build model
        self.model = AgeGenderBackbone(input_size=self.input_size, width_mult=1.0)

        # Load weights
        try:
            weights_path = get_weights_path(model_name, backend='mlx')
            load_mlx_weights(self.model, weights_path)
            Logger.info(f'Loaded MLX weights from {weights_path}')
        except (ValueError, FileNotFoundError) as e:
            Logger.warning(f'MLX weights not available: {e}. Model initialized without weights.')

        # Set to inference mode
        self.model.train(False)

    def _initialize_model(self) -> None:
        """Initialize model - already done in __init__."""
        pass

    def preprocess(self, image: np.ndarray, bbox: Union[List, np.ndarray]) -> mx.array:
        """
        Align face and prepare for inference.

        Args:
            image: Full input image in BGR format.
            bbox: Face bounding box [x1, y1, x2, y2].

        Returns:
            Preprocessed MLX array.
        """
        bbox = np.asarray(bbox)

        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        scale = self.input_size[1] / (max(width, height) * 1.5)

        aligned_face, _ = bbox_center_alignment(image, center, self.input_size[1], scale, 0.0)

        # Convert BGR to RGB
        rgb_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Resize if needed
        if rgb_face.shape[:2] != self.input_size:
            rgb_face = cv2.resize(rgb_face, self.input_size[::-1])

        # Add batch dimension (H, W, C) -> (1, H, W, C)
        batch_face = np.expand_dims(rgb_face, axis=0)

        return mx.array(batch_face)

    def inference(self, input_tensor: mx.array) -> mx.array:
        """Perform MLX inference."""
        output = self.model(input_tensor)
        synchronize(output)
        return output

    def postprocess(self, prediction: np.ndarray) -> Tuple[int, int]:
        """
        Process raw output to get gender and age.

        Args:
            prediction: Raw model output [gender_0, gender_1, age].

        Returns:
            Tuple of (gender_id, age).
        """
        # First two values are gender logits
        gender_id = int(np.argmax(prediction[:2]))
        # Third value is normalized age, scaled by 100
        age = int(np.round(prediction[2] * 100))
        return gender_id, age

    def predict(self, image: np.ndarray, bbox: Union[List, np.ndarray]) -> Tuple[int, int]:
        """
        Predict age and gender for a face.

        Args:
            image: Full input image in BGR format.
            bbox: Face bounding box [x1, y1, x2, y2].

        Returns:
            Tuple of (gender_id, age) where gender_id is 0=Female, 1=Male.
        """
        # Preprocess
        input_tensor = self.preprocess(image, bbox)

        # Inference
        mlx_output = self.inference(input_tensor)

        # Convert to numpy
        prediction = to_numpy(mlx_output)[0]

        # Postprocess
        gender_id, age = self.postprocess(prediction)

        return gender_id, age
