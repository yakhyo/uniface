# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo
#
# SCRFD MLX Implementation

"""
SCRFD face detector implemented in MLX for Apple Silicon.

SCRFD (Sample and Computation Redistribution for Efficient Face Detection)
is a high-efficiency face detector. This MLX implementation provides
blazing fast inference on M1/M2/M3/M4 chips.
"""

from typing import Any, Dict, List, Literal, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from uniface.common import distance2bbox, distance2kps, non_max_suppression, resize_image
from uniface.constants import SCRFDWeights
from uniface.detection.base import BaseDetector
from uniface.log import Logger
from uniface.mlx_utils import load_mlx_weights, synchronize, to_numpy
from uniface.model_store import get_weights_path
from uniface.nn.backbone import MobileNetV2
from uniface.nn.conv import ConvBNReLU
from uniface.nn.fpn import FPN

__all__ = ['SCRFDMLX']


class SCRFDHead(nn.Module):
    """
    SCRFD detection head for a single FPN level.

    Predicts scores, bboxes, and keypoints for each anchor.
    """

    def __init__(self, in_channels: int, num_anchors: int = 2):
        super().__init__()
        self.num_anchors = num_anchors

        # Shared conv
        self.conv = ConvBNReLU(in_channels, in_channels, kernel_size=3, padding=1)

        # Score head (1 value per anchor - binary classification)
        self.score_conv = nn.Conv2d(in_channels, num_anchors, kernel_size=1)

        # BBox head (4 values per anchor - distances to edges)
        self.bbox_conv = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

        # Keypoint head (10 values per anchor - 5 points x 2 coords)
        self.kps_conv = nn.Conv2d(in_channels, num_anchors * 10, kernel_size=1)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Forward pass returning scores, bbox preds, and keypoint preds."""
        x = self.conv(x)

        scores = mx.sigmoid(self.score_conv(x))
        bbox_preds = self.bbox_conv(x)
        kps_preds = self.kps_conv(x)

        return scores, bbox_preds, kps_preds


class SCRFDNetwork(nn.Module):
    """
    SCRFD neural network architecture in MLX.

    Architecture:
    - Backbone: MobileNetV2 (or similar efficient backbone)
    - Neck: Simple FPN
    - Heads: Per-level detection heads
    """

    def __init__(self, num_anchors: int = 2):
        super().__init__()

        # Backbone - using MobileNetV2 for efficiency
        self.backbone = MobileNetV2(width_mult=1.0, return_stages=[3, 5, 7])
        in_channels_list = self.backbone.get_out_channels([3, 5, 7])

        # FPN
        fpn_out_channels = 64
        self.fpn = FPN(in_channels_list=in_channels_list, out_channels=fpn_out_channels)

        # Detection heads (one per FPN level)
        self.head1 = SCRFDHead(fpn_out_channels, num_anchors)
        self.head2 = SCRFDHead(fpn_out_channels, num_anchors)
        self.head3 = SCRFDHead(fpn_out_channels, num_anchors)

    def __call__(self, x: mx.array) -> List[Tuple[mx.array, mx.array, mx.array]]:
        """
        Forward pass.

        Args:
            x: Input tensor (N, H, W, 3) in NHWC format.

        Returns:
            List of (scores, bbox_preds, kps_preds) tuples, one per FPN level.
        """
        # Backbone features
        features = self.backbone(x)

        # FPN fusion
        fpn_features = self.fpn(features)

        # Apply heads
        out1 = self.head1(fpn_features[0])
        out2 = self.head2(fpn_features[1])
        out3 = self.head3(fpn_features[2])

        return [out1, out2, out3]


class SCRFDMLX(BaseDetector):
    """
    SCRFD face detector using MLX backend for Apple Silicon.

    API is identical to the ONNX version for seamless switching.

    Args:
        model_name: Model weights to use. Defaults to SCRFD_10G_KPS.
        conf_thresh: Confidence threshold. Defaults to 0.5.
        nms_thresh: NMS IoU threshold. Defaults to 0.4.
        input_size: Input size (width, height). Defaults to (640, 640).
    """

    def __init__(
        self,
        *,
        model_name: SCRFDWeights = SCRFDWeights.SCRFD_10G_KPS,
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

        # SCRFD model params
        self._fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self._center_cache = {}

        Logger.info(
            f'Initializing SCRFD (MLX) with model={self.model_name}, '
            f'conf_thresh={self.conf_thresh}, nms_thresh={self.nms_thresh}'
        )

        # Build network
        self.model = SCRFDNetwork(num_anchors=self._num_anchors)

        # Load weights
        try:
            weights_path = get_weights_path(model_name, backend='mlx')
            load_mlx_weights(self.model, weights_path)
            Logger.info(f'Loaded MLX weights from {weights_path}')
        except (ValueError, FileNotFoundError) as e:
            Logger.warning(f'MLX weights not available: {e}. Model initialized without weights.')

        # Set to inference mode
        self.model.train(False)

        # Compile the model forward pass for better performance
        self._compiled_forward = mx.compile(self.model)
        Logger.debug('Compiled model forward pass with mx.compile')

    def preprocess(self, image: np.ndarray) -> mx.array:
        """Preprocess image for MLX inference."""
        image = image.astype(np.float32)
        image = (image - 127.5) / 127.5

        # Add batch dimension (H, W, C) -> (1, H, W, C)
        image = np.expand_dims(image, axis=0)

        return mx.array(image)

    def inference(self, input_tensor: mx.array) -> List[Tuple[mx.array, mx.array, mx.array]]:
        """Perform MLX inference using compiled model."""
        outputs = self._compiled_forward(input_tensor)

        # Force computation
        for scores, bboxes, kps in outputs:
            synchronize(scores, bboxes, kps)

        return outputs

    def postprocess(
        self,
        outputs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        image_size: Tuple[int, int],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Process model outputs into detections."""
        scores_list = []
        bboxes_list = []
        kpss_list = []

        for idx, stride in enumerate(self._feat_stride_fpn):
            scores, bbox_preds, kps_preds = outputs[idx]

            # Squeeze batch dimension and convert to numpy
            scores = scores.squeeze(0)  # (H, W, num_anchors)
            bbox_preds = bbox_preds.squeeze(0) * stride
            kps_preds = kps_preds.squeeze(0) * stride

            # Reshape for processing
            fm_height, fm_width = scores.shape[:2]
            scores = scores.reshape(-1)
            bbox_preds = bbox_preds.reshape(-1, 4)
            kps_preds = kps_preds.reshape(-1, 10)

            # Generate anchor centers
            cache_key = (fm_height, fm_width, stride)
            if cache_key in self._center_cache:
                anchor_centers = self._center_cache[cache_key]
            else:
                y, x = np.mgrid[:fm_height, :fm_width]
                anchor_centers = np.stack((x, y), axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape(-1, 2)

                if self._num_anchors > 1:
                    anchor_centers = np.tile(anchor_centers[:, None, :], (1, self._num_anchors, 1)).reshape(-1, 2)

                if len(self._center_cache) < 100:
                    self._center_cache[cache_key] = anchor_centers

            # Filter by confidence
            pos_indices = np.where(scores >= self.conf_thresh)[0]
            if len(pos_indices) == 0:
                continue

            bboxes = distance2bbox(anchor_centers, bbox_preds)[pos_indices]
            scores_selected = scores[pos_indices]
            scores_list.append(scores_selected[:, None])
            bboxes_list.append(bboxes)

            landmarks = distance2kps(anchor_centers, kps_preds)
            landmarks = landmarks.reshape((landmarks.shape[0], -1, 2))
            kpss_list.append(landmarks[pos_indices])

        return scores_list, bboxes_list, kpss_list

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
            image: Input image (H, W, C) in BGR format.
            max_num: Maximum detections (0 = all).
            metric: Ranking metric.
            center_weight: Weight for center-based ranking.

        Returns:
            List of face dictionaries with 'bbox', 'confidence', 'landmarks'.
        """
        original_height, original_width = image.shape[:2]

        image, resize_factor = resize_image(image, target_shape=self.input_size)
        input_tensor = self.preprocess(image)

        # MLX inference
        mlx_outputs = self.inference(input_tensor)

        # Convert to numpy for postprocessing
        outputs = []
        for scores, bboxes, kps in mlx_outputs:
            outputs.append(
                (
                    to_numpy(scores),
                    to_numpy(bboxes),
                    to_numpy(kps),
                )
            )

        scores_list, bboxes_list, kpss_list = self.postprocess(outputs, image_size=image.shape[:2])

        # Handle no detections
        if not scores_list:
            return []

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        bboxes = np.vstack(bboxes_list) / resize_factor
        landmarks = np.vstack(kpss_list) / resize_factor

        pre_det = np.hstack((bboxes, scores)).astype(np.float32)
        pre_det = pre_det[order, :]

        keep = non_max_suppression(pre_det, threshold=self.nms_thresh)

        detections = pre_det[keep, :]
        landmarks = landmarks[order, :, :]
        landmarks = landmarks[keep, :, :].astype(np.float32)

        # Apply max_num filtering
        if 0 < max_num < detections.shape[0]:
            area = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
            center = (original_height // 2, original_width // 2)
            offsets = np.vstack(
                [
                    (detections[:, 0] + detections[:, 2]) / 2 - center[1],
                    (detections[:, 1] + detections[:, 3]) / 2 - center[0],
                ]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), axis=0)

            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * center_weight

            sorted_indices = np.argsort(values)[::-1][:max_num]
            detections = detections[sorted_indices]
            landmarks = landmarks[sorted_indices]

        faces = []
        for i in range(detections.shape[0]):
            face_dict = {
                'bbox': detections[i, :4],
                'confidence': float(detections[i, 4]),
                'landmarks': landmarks[i],
            }
            faces.append(face_dict)

        return faces
