# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo
#
# RetinaFace MLX Implementation

"""
RetinaFace face detector implemented in MLX for Apple Silicon.

This module provides an MLX-native implementation of RetinaFace that runs
blazingly fast on M1/M2/M3/M4 chips using Apple's unified memory architecture.
"""

from typing import Any, Dict, List, Literal, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from uniface.common import (
    decode_boxes,
    decode_landmarks,
    generate_anchors,
    non_max_suppression,
    resize_image,
)
from uniface.constants import RetinaFaceWeights
from uniface.detection.base import BaseDetector
from uniface.log import Logger
from uniface.mlx_utils import load_mlx_weights, synchronize, to_numpy
from uniface.model_store import get_weights_path
from uniface.nn.backbone import MobileNetV1, MobileNetV2, MobileNetV2Fused
from uniface.nn.fpn import FPN, FPNFused, SSH, SSHFused
from uniface.nn.head import BboxHeadWrapper, ClassHeadWrapper, LandmarkHeadWrapper

__all__ = ['RetinaFaceMLX']


class RetinaFaceNetworkFused(nn.Module):
    """
    RetinaFace network with fused BatchNorm (for ONNX-converted weights).

    This variant uses fused Conv layers (no separate BatchNorm) for loading
    weights from ONNX models where BatchNorm has been fused into Conv during
    export optimization. This achieves numerical parity with ONNX inference.

    Architecture:
    - Backbone: MobileNetV2Fused
    - Neck: FPNFused with SSHFused context modules
    - Heads: Classification, BBox regression, Landmark regression (these don't
             have BatchNorm in the original, so they stay the same)
    """

    def __init__(
        self,
        backbone_type: str = 'mobilenetv2',
        width_mult: float = 1.0,
        num_anchors: int = 2,
    ):
        super().__init__()

        # Build fused backbone
        if backbone_type == 'mobilenetv2':
            self.backbone = MobileNetV2Fused(width_mult=width_mult, return_features=[6, 13, 18])
            in_channels_list = self.backbone.get_out_channels()
        else:
            raise ValueError(f'Fused backbone only supports mobilenetv2, got: {backbone_type}')

        # FPN output channels (128 to match PyTorch RetinaFace)
        fpn_out_channels = 128

        # Build fused FPN
        self.fpn = FPNFused(in_channels_list=in_channels_list, out_channels=fpn_out_channels)

        # Build fused SSH context modules
        self.ssh1 = SSHFused(fpn_out_channels, fpn_out_channels)
        self.ssh2 = SSHFused(fpn_out_channels, fpn_out_channels)
        self.ssh3 = SSHFused(fpn_out_channels, fpn_out_channels)

        # Detection heads (no BatchNorm in original, so same as unfused)
        self.class_head = ClassHeadWrapper(fpn_out_channels, num_anchors, num_classes=2)
        self.bbox_head = BboxHeadWrapper(fpn_out_channels, num_anchors)
        self.landmark_head = LandmarkHeadWrapper(fpn_out_channels, num_anchors, num_landmarks=5)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Forward pass."""
        # Extract backbone features
        features = self.backbone(x)

        # FPN fusion
        fpn_features = self.fpn(features)

        # Apply SSH context modules
        f1 = self.ssh1(fpn_features[0])
        f2 = self.ssh2(fpn_features[1])
        f3 = self.ssh3(fpn_features[2])

        # Apply detection heads
        cls1 = self.class_head.forward_level(f1, 0)
        cls2 = self.class_head.forward_level(f2, 1)
        cls3 = self.class_head.forward_level(f3, 2)

        bbox1 = self.bbox_head.forward_level(f1, 0)
        bbox2 = self.bbox_head.forward_level(f2, 1)
        bbox3 = self.bbox_head.forward_level(f3, 2)

        lmk1 = self.landmark_head.forward_level(f1, 0)
        lmk2 = self.landmark_head.forward_level(f2, 1)
        lmk3 = self.landmark_head.forward_level(f3, 2)

        # Concatenate predictions from all levels
        cls_preds = mx.concatenate([cls1, cls2, cls3], axis=1)
        bbox_preds = mx.concatenate([bbox1, bbox2, bbox3], axis=1)
        landmark_preds = mx.concatenate([lmk1, lmk2, lmk3], axis=1)

        return cls_preds, bbox_preds, landmark_preds


class RetinaFaceNetwork(nn.Module):
    """
    RetinaFace neural network architecture in MLX.

    Architecture:
    - Backbone: MobileNetV1 or MobileNetV2
    - Neck: FPN with SSH context modules
    - Heads: Classification, BBox regression, Landmark regression
    """

    def __init__(
        self,
        backbone_type: str = 'mobilenetv2',
        width_mult: float = 1.0,
        num_anchors: int = 2,
    ):
        super().__init__()

        # Build backbone
        if backbone_type == 'mobilenetv1':
            self.backbone = MobileNetV1(width_mult=width_mult, return_stages=[2, 3, 4])
            in_channels_list = self.backbone.get_out_channels([2, 3, 4])
        elif backbone_type == 'mobilenetv2':
            # Use return_features for feature index-based extraction
            # RetinaFace uses features.6 (32ch), features.13 (96ch), features.18 (1280ch)
            self.backbone = MobileNetV2(width_mult=width_mult, return_features=[6, 13, 18])
            in_channels_list = self.backbone.get_out_channels()
        else:
            raise ValueError(f'Unknown backbone type: {backbone_type}')

        # FPN output channels (128 to match PyTorch RetinaFace)
        fpn_out_channels = 128

        # Build FPN
        self.fpn = FPN(in_channels_list=in_channels_list, out_channels=fpn_out_channels)

        # Build SSH context modules (one per FPN level)
        self.ssh1 = SSH(fpn_out_channels, fpn_out_channels)
        self.ssh2 = SSH(fpn_out_channels, fpn_out_channels)
        self.ssh3 = SSH(fpn_out_channels, fpn_out_channels)

        # Build detection heads (wrapper structure matching PyTorch naming)
        self.class_head = ClassHeadWrapper(fpn_out_channels, num_anchors, num_classes=2)
        self.bbox_head = BboxHeadWrapper(fpn_out_channels, num_anchors)
        self.landmark_head = LandmarkHeadWrapper(fpn_out_channels, num_anchors, num_landmarks=5)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (N, H, W, 3) in NHWC format.

        Returns:
            Tuple of (cls_scores, bbox_preds, landmark_preds).
        """
        # Extract backbone features
        features = self.backbone(x)  # [C3, C4, C5] or equivalent

        # FPN fusion
        fpn_features = self.fpn(features)  # [P3, P4, P5]

        # Apply SSH context modules
        f1 = self.ssh1(fpn_features[0])
        f2 = self.ssh2(fpn_features[1])
        f3 = self.ssh3(fpn_features[2])

        # Apply detection heads (using wrapper's level-specific forward)
        cls1 = self.class_head.forward_level(f1, 0)
        cls2 = self.class_head.forward_level(f2, 1)
        cls3 = self.class_head.forward_level(f3, 2)

        bbox1 = self.bbox_head.forward_level(f1, 0)
        bbox2 = self.bbox_head.forward_level(f2, 1)
        bbox3 = self.bbox_head.forward_level(f3, 2)

        lmk1 = self.landmark_head.forward_level(f1, 0)
        lmk2 = self.landmark_head.forward_level(f2, 1)
        lmk3 = self.landmark_head.forward_level(f3, 2)

        # Concatenate predictions from all levels
        cls_preds = mx.concatenate([cls1, cls2, cls3], axis=1)
        bbox_preds = mx.concatenate([bbox1, bbox2, bbox3], axis=1)
        landmark_preds = mx.concatenate([lmk1, lmk2, lmk3], axis=1)

        return cls_preds, bbox_preds, landmark_preds


class RetinaFaceMLX(BaseDetector):
    """
    RetinaFace face detector using MLX backend for Apple Silicon.

    This is the MLX-native implementation that provides blazing fast inference
    on M1/M2/M3/M4 chips. API is identical to the ONNX version.

    Args:
        model_name: Model weights to use. Defaults to RetinaFaceWeights.MNET_V2.
        conf_thresh: Confidence threshold. Defaults to 0.5.
        nms_thresh: NMS IoU threshold. Defaults to 0.4.
        input_size: Input size (width, height). Defaults to (640, 640).
        **kwargs: Additional options (pre_nms_topk, post_nms_topk, dynamic_size).
    """

    def __init__(
        self,
        *,
        model_name: RetinaFaceWeights = RetinaFaceWeights.MNET_V2,
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        input_size: Tuple[int, int] = (640, 640),
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            conf_thresh=conf_thresh,
            nms_thresh=nms_thresh,
            input_size=input_size,
            **kwargs,
        )
        self._supports_landmarks = True

        self.model_name = model_name
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size

        # Advanced options
        self.pre_nms_topk = kwargs.get('pre_nms_topk', 5000)
        self.post_nms_topk = kwargs.get('post_nms_topk', 750)
        self.dynamic_size = kwargs.get('dynamic_size', False)

        Logger.info(
            f'Initializing RetinaFace (MLX) with model={self.model_name}, '
            f'conf_thresh={self.conf_thresh}, nms_thresh={self.nms_thresh}'
        )

        # Determine backbone type and width from model name
        self._backbone_type, self._width_mult = self._get_backbone_config(model_name)

        # Build the network
        self.model = RetinaFaceNetwork(
            backbone_type=self._backbone_type,
            width_mult=self._width_mult,
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

        # Precompute anchors if using static size
        if not self.dynamic_size and self.input_size is not None:
            self._priors = generate_anchors(image_size=self.input_size)
            Logger.debug('Generated anchors for static input size.')

    def _get_backbone_config(self, model_name: RetinaFaceWeights) -> Tuple[str, float]:
        """Get backbone type and width multiplier from model name."""
        config_map = {
            RetinaFaceWeights.MNET_025: ('mobilenetv1', 0.25),
            RetinaFaceWeights.MNET_050: ('mobilenetv1', 0.5),
            RetinaFaceWeights.MNET_V1: ('mobilenetv1', 1.0),
            RetinaFaceWeights.MNET_V2: ('mobilenetv2', 1.0),
            RetinaFaceWeights.RESNET18: ('mobilenetv2', 1.0),  # TODO: Implement ResNet
            RetinaFaceWeights.RESNET34: ('mobilenetv2', 1.0),  # TODO: Implement ResNet
        }
        return config_map.get(model_name, ('mobilenetv2', 1.0))

    def preprocess(self, image: np.ndarray) -> mx.array:
        """
        Preprocess input image for MLX inference.

        Args:
            image: Input image in BGR format (H, W, C).

        Returns:
            MLX array in NHWC format ready for inference.
        """
        # Subtract mean (BGR order)
        image = np.float32(image) - np.array([104, 117, 123], dtype=np.float32)

        # Add batch dimension: (H, W, C) -> (1, H, W, C)
        image = np.expand_dims(image, axis=0)

        # Convert to MLX array
        return mx.array(image)

    def inference(self, input_tensor: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Perform MLX inference.

        Args:
            input_tensor: Preprocessed input tensor in NHWC format.

        Returns:
            Tuple of (cls_scores, bbox_preds, landmark_preds).
        """
        cls_preds, bbox_preds, landmark_preds = self.model(input_tensor)

        # Force computation (MLX uses lazy evaluation)
        synchronize(cls_preds, bbox_preds, landmark_preds)

        return cls_preds, bbox_preds, landmark_preds

    def detect(
        self,
        image: np.ndarray,
        *,
        max_num: int = 0,
        metric: Literal['default', 'max'] = 'max',
        center_weight: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """
        Detect faces in an image.

        Args:
            image: Input image as NumPy array (H, W, C) in BGR format.
            max_num: Maximum detections to return (0 = all).
            metric: Ranking metric ('default' or 'max').
            center_weight: Weight for center-based ranking.

        Returns:
            List of face dictionaries with 'bbox', 'confidence', 'landmarks'.
        """
        original_height, original_width = image.shape[:2]

        if self.dynamic_size:
            height, width, _ = image.shape
            self._priors = generate_anchors(image_size=(height, width))
            resize_factor = 1.0
        else:
            image, resize_factor = resize_image(image, target_shape=self.input_size)

        height, width, _ = image.shape
        input_tensor = self.preprocess(image)

        # MLX inference
        cls_preds, bbox_preds, landmark_preds = self.inference(input_tensor)

        # Apply softmax to classification logits to get probabilities
        # ONNX model has softmax built-in, MLX model outputs raw logits
        cls_probs = mx.softmax(cls_preds, axis=-1)

        # Convert to numpy for postprocessing
        outputs = [
            to_numpy(bbox_preds),
            to_numpy(cls_probs),
            to_numpy(landmark_preds),
        ]

        # Postprocessing (reuse existing NumPy-based logic)
        detections, landmarks = self.postprocess(outputs, resize_factor, shape=(width, height))

        # Apply max_num filtering
        if max_num > 0 and detections.shape[0] > max_num:
            areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
            center = (original_height // 2, original_width // 2)
            offsets = np.vstack(
                [
                    (detections[:, 0] + detections[:, 2]) / 2 - center[1],
                    (detections[:, 1] + detections[:, 3]) / 2 - center[0],
                ]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), axis=0)

            if metric == 'max':
                scores = areas
            else:
                scores = areas - offset_dist_squared * center_weight

            sorted_indices = np.argsort(scores)[::-1][:max_num]
            detections = detections[sorted_indices]
            landmarks = landmarks[sorted_indices]

        # Build output
        faces = []
        for i in range(detections.shape[0]):
            face_dict = {
                'bbox': detections[i, :4],
                'confidence': float(detections[i, 4]),
                'landmarks': landmarks[i],
            }
            faces.append(face_dict)

        return faces

    def postprocess(
        self,
        outputs: List[np.ndarray],
        resize_factor: float,
        shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process model outputs into final detections.

        Args:
            outputs: [bbox_preds, cls_preds, landmark_preds].
            resize_factor: Image resize factor.
            shape: (width, height) of processed image.

        Returns:
            Tuple of (detections, landmarks).
        """
        loc = outputs[0].squeeze(0)
        conf = outputs[1].squeeze(0)
        landmarks = outputs[2].squeeze(0)

        # Decode boxes and landmarks
        boxes = decode_boxes(loc, self._priors)
        landmarks = decode_landmarks(landmarks, self._priors)

        boxes, landmarks = self._scale_detections(boxes, landmarks, resize_factor, shape)

        # Extract face class scores
        scores = conf[:, 1]
        mask = scores > self.conf_thresh

        boxes, landmarks, scores = boxes[mask], landmarks[mask], scores[mask]

        # Sort by score
        order = scores.argsort()[::-1][: self.pre_nms_topk]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

        # Apply NMS
        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
        keep = non_max_suppression(detections, self.nms_thresh)
        detections, landmarks = detections[keep], landmarks[keep]

        # Keep top-k
        detections = detections[: self.post_nms_topk]
        landmarks = landmarks[: self.post_nms_topk]

        landmarks = landmarks.reshape(-1, 5, 2).astype(np.float32)

        return detections, landmarks

    def _scale_detections(
        self,
        boxes: np.ndarray,
        landmarks: np.ndarray,
        resize_factor: float,
        shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scale detections to original image size."""
        bbox_scale = np.array([shape[0], shape[1]] * 2)
        boxes = boxes * bbox_scale / resize_factor

        landmark_scale = np.array([shape[0], shape[1]] * 5)
        landmarks = landmarks * landmark_scale / resize_factor

        return boxes, landmarks
