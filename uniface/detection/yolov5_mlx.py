# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo
#
# YOLOv5Face MLX Implementation

"""
YOLOv5-Face detector implemented in MLX for Apple Silicon.

YOLOv5-Face is an accurate face detector based on the YOLOv5 architecture,
with support for 5-point facial landmarks. This MLX implementation provides
fast inference on M1/M2/M3/M4 chips.

Paper: "YOLO5Face: Why Reinventing a Face Detector"
       https://arxiv.org/abs/2105.12931
"""

from typing import Any, Dict, List, Literal, Tuple

import cv2
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from uniface.common import non_max_suppression
from uniface.constants import YOLOv5FaceWeights
from uniface.detection.base import BaseDetector
from uniface.log import Logger
from uniface.mlx_utils import load_mlx_weights, synchronize, to_numpy
from uniface.model_store import get_weights_path
from uniface.nn.conv import C3, SPPF, Concat, ConvBNSiLU

__all__ = ['YOLOv5FaceMLX']


class YOLOv5Backbone(nn.Module):
    """
    YOLOv5 CSPDarknet backbone.

    This is the feature extraction backbone using CSP (Cross Stage Partial)
    blocks with SiLU activation.

    Architecture (YOLOv5s):
    - P1: Conv(3, 32, k=6, s=2) -> Conv(32, 64, k=3, s=2) -> C3(64, 64, n=1)
    - P2: Conv(64, 128, k=3, s=2) -> C3(128, 128, n=2)
    - P3: Conv(128, 256, k=3, s=2) -> C3(256, 256, n=3) [stride 8]
    - P4: Conv(256, 512, k=3, s=2) -> C3(512, 512, n=1) [stride 16]
    - P5: Conv(512, 1024, k=3, s=2) -> C3(1024, 1024, n=1) -> SPPF [stride 32]

    Args:
        width_mult: Width multiplier for channel scaling. Default: 0.5 (for yolov5s).
        depth_mult: Depth multiplier for layer count scaling. Default: 0.33.
    """

    def __init__(
        self,
        width_mult: float = 0.5,
        depth_mult: float = 0.33,
    ):
        super().__init__()

        def ch(c: int) -> int:
            """Apply width multiplier."""
            return max(int(c * width_mult), 8)

        def depth(d: int) -> int:
            """Apply depth multiplier."""
            return max(int(d * depth_mult), 1)

        # Stage 0: Focus layer (replaced with 6x6 conv in newer versions)
        self.stem = ConvBNSiLU(3, ch(64), kernel_size=6, stride=2, padding=2)

        # Stage 1: P1
        self.stage1 = nn.Sequential(
            ConvBNSiLU(ch(64), ch(128), kernel_size=3, stride=2, padding=1),
            C3(ch(128), ch(128), n=depth(3)),
        )

        # Stage 2: P2 (output stride 8)
        self.stage2 = nn.Sequential(
            ConvBNSiLU(ch(128), ch(256), kernel_size=3, stride=2, padding=1),
            C3(ch(256), ch(256), n=depth(6)),
        )

        # Stage 3: P3 (output stride 16)
        self.stage3 = nn.Sequential(
            ConvBNSiLU(ch(256), ch(512), kernel_size=3, stride=2, padding=1),
            C3(ch(512), ch(512), n=depth(9)),
        )

        # Stage 4: P4 (output stride 32)
        self.stage4 = nn.Sequential(
            ConvBNSiLU(ch(512), ch(1024), kernel_size=3, stride=2, padding=1),
            C3(ch(1024), ch(1024), n=depth(3)),
            SPPF(ch(1024), ch(1024)),
        )

        # Store output channels for FPN
        self.out_channels = [ch(256), ch(512), ch(1024)]

    def __call__(self, x: mx.array) -> List[mx.array]:
        """
        Forward pass returning multi-scale features.

        Args:
            x: Input tensor of shape (N, H, W, 3) in NHWC format.

        Returns:
            List of [P3, P4, P5] feature maps.
        """
        x = self.stem(x)
        x = self.stage1(x)

        # P3: stride 8
        p3 = self.stage2(x)

        # P4: stride 16
        p4 = self.stage3(p3)

        # P5: stride 32
        p5 = self.stage4(p4)

        return [p3, p4, p5]


class YOLOv5Neck(nn.Module):
    """
    YOLOv5 PANet (Path Aggregation Network) neck.

    This combines top-down (FPN) and bottom-up feature fusion
    for better multi-scale detection.

    Architecture:
    - Top-down path: P5 -> P4 -> P3 (upsampling + concat + C3)
    - Bottom-up path: P3 -> P4 -> P5 (downsampling + concat + C3)
    """

    def __init__(
        self,
        in_channels: List[int],
        width_mult: float = 0.5,
        depth_mult: float = 0.33,
    ):
        super().__init__()

        def ch(c: int) -> int:
            return max(int(c * width_mult), 8)

        def depth(d: int) -> int:
            return max(int(d * depth_mult), 1)

        c3, c4, c5 = in_channels  # [256, 512, 1024] scaled

        # Top-down path
        self.lateral_conv1 = ConvBNSiLU(c5, c4, kernel_size=1)
        self.c3_p4 = C3(c4 * 2, c4, n=depth(3), shortcut=False)

        self.lateral_conv2 = ConvBNSiLU(c4, c3, kernel_size=1)
        self.c3_p3 = C3(c3 * 2, c3, n=depth(3), shortcut=False)

        # Bottom-up path
        self.down_conv1 = ConvBNSiLU(c3, c3, kernel_size=3, stride=2, padding=1)
        self.c3_n3 = C3(c3 * 2, c4, n=depth(3), shortcut=False)

        self.down_conv2 = ConvBNSiLU(c4, c4, kernel_size=3, stride=2, padding=1)
        self.c3_n4 = C3(c4 * 2, c5, n=depth(3), shortcut=False)

        self.concat = Concat()

    def __call__(self, features: List[mx.array]) -> List[mx.array]:
        """
        Forward pass for PANet feature fusion.

        Args:
            features: [P3, P4, P5] from backbone.

        Returns:
            [N3, N4, N5] fused features for detection heads.
        """
        p3, p4, p5 = features

        # Top-down path
        # P5 -> upsample -> concat with P4
        p5_up = self.lateral_conv1(p5)
        p5_up = self._upsample(p5_up, p4.shape[1:3])
        p4_td = self.c3_p4(self.concat([p5_up, p4]))

        # P4 -> upsample -> concat with P3
        p4_up = self.lateral_conv2(p4_td)
        p4_up = self._upsample(p4_up, p3.shape[1:3])
        n3 = self.c3_p3(self.concat([p4_up, p3]))

        # Bottom-up path
        # N3 -> downsample -> concat with P4_td
        n3_down = self.down_conv1(n3)
        n4 = self.c3_n3(self.concat([n3_down, p4_td]))

        # N4 -> downsample -> concat with P5
        n4_down = self.down_conv2(n4)
        n5 = self.c3_n4(self.concat([n4_down, p5]))

        return [n3, n4, n5]

    def _upsample(self, x: mx.array, target_size: Tuple[int, int]) -> mx.array:
        """Upsample feature map to target size using nearest neighbor."""
        n, h, w, c = x.shape
        target_h, target_w = target_size

        # Nearest neighbor upsampling
        # Repeat each element along H and W dimensions
        scale_h = target_h // h
        scale_w = target_w // w

        # Use repeat for upsampling
        x = mx.repeat(x, scale_h, axis=1)
        x = mx.repeat(x, scale_w, axis=2)

        return x


class YOLOv5Head(nn.Module):
    """
    YOLOv5-Face detection head.

    Outputs predictions for each anchor at each scale:
    - 4 bbox coordinates (x, y, w, h)
    - 1 objectness score
    - 1 class confidence (face only)
    - 10 landmark coordinates (5 points x 2)

    Total: 16 values per anchor.
    """

    def __init__(
        self,
        in_channels: List[int],
        num_anchors: int = 3,
    ):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_outputs = 16  # 4 + 1 + 1 + 10 (bbox + obj + cls + landmarks)

        # Detection heads for each scale
        self.head1 = nn.Conv2d(in_channels[0], num_anchors * self.num_outputs, kernel_size=1)
        self.head2 = nn.Conv2d(in_channels[1], num_anchors * self.num_outputs, kernel_size=1)
        self.head3 = nn.Conv2d(in_channels[2], num_anchors * self.num_outputs, kernel_size=1)

    def __call__(self, features: List[mx.array]) -> List[mx.array]:
        """
        Apply detection heads to features.

        Args:
            features: [N3, N4, N5] from neck.

        Returns:
            List of detection outputs, one per scale.
        """
        out1 = self.head1(features[0])  # stride 8
        out2 = self.head2(features[1])  # stride 16
        out3 = self.head3(features[2])  # stride 32

        return [out1, out2, out3]


class YOLOv5FaceNetwork(nn.Module):
    """
    Complete YOLOv5-Face network architecture.

    Combines backbone, neck (PANet), and detection heads.
    """

    def __init__(
        self,
        width_mult: float = 0.5,
        depth_mult: float = 0.33,
        num_anchors: int = 3,
    ):
        super().__init__()

        # Backbone
        self.backbone = YOLOv5Backbone(width_mult=width_mult, depth_mult=depth_mult)

        # Neck
        self.neck = YOLOv5Neck(
            in_channels=self.backbone.out_channels,
            width_mult=width_mult,
            depth_mult=depth_mult,
        )

        # Head
        # After neck, channels are [c3, c4, c5] which equals backbone.out_channels
        self.head = YOLOv5Head(
            in_channels=self.backbone.out_channels,
            num_anchors=num_anchors,
        )

        self.strides = [8, 16, 32]

    def __call__(self, x: mx.array) -> List[mx.array]:
        """
        Forward pass.

        Args:
            x: Input tensor (N, H, W, 3) in NHWC format.

        Returns:
            List of detection outputs per scale.
        """
        features = self.backbone(x)
        features = self.neck(features)
        outputs = self.head(features)
        return outputs


class YOLOv5FaceMLX(BaseDetector):
    """
    YOLOv5-Face detector using MLX backend for Apple Silicon.

    API is identical to the ONNX version for seamless switching.

    Args:
        model_name: Model weights to use. Defaults to YOLOV5S.
        conf_thresh: Confidence threshold. Defaults to 0.6.
        nms_thresh: NMS IoU threshold. Defaults to 0.5.
        input_size: Input image size (must be 640). Defaults to 640.
        **kwargs: Additional options (max_det).
    """

    def __init__(
        self,
        *,
        model_name: YOLOv5FaceWeights = YOLOv5FaceWeights.YOLOV5S,
        conf_thresh: float = 0.6,
        nms_thresh: float = 0.5,
        input_size: int = 640,
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

        # Validate input size
        if input_size != 640:
            raise ValueError(
                f'YOLOv5Face only supports input_size=640 (got {input_size}). The model has a fixed input shape.'
            )

        self.model_name = model_name
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size

        # Advanced options
        self.max_det = kwargs.get('max_det', 750)

        Logger.info(
            f'Initializing YOLOv5Face (MLX) with model={self.model_name}, '
            f'conf_thresh={self.conf_thresh}, nms_thresh={self.nms_thresh}'
        )

        # Get model configuration based on variant
        width_mult, depth_mult = self._get_model_config(model_name)

        # Build the network
        self.model = YOLOv5FaceNetwork(
            width_mult=width_mult,
            depth_mult=depth_mult,
        )

        # Anchor configuration for YOLOv5-Face
        self.anchors = np.array(
            [
                [[4, 5], [8, 10], [13, 16]],  # stride 8
                [[23, 29], [43, 55], [73, 105]],  # stride 16
                [[146, 217], [231, 300], [335, 433]],  # stride 32
            ],
            dtype=np.float32,
        )
        self.strides = np.array([8, 16, 32])

        # Load weights
        try:
            weights_path = get_weights_path(model_name, backend='mlx')
            load_mlx_weights(self.model, weights_path)
            Logger.info(f'Loaded MLX weights from {weights_path}')
        except (ValueError, FileNotFoundError) as e:
            Logger.warning(f'MLX weights not available: {e}. Model initialized without weights.')

        # Set to inference mode
        self.model.train(False)

        # Precompute grid and anchor grids for each scale
        self._init_grids()

    def _get_model_config(self, model_name: YOLOv5FaceWeights) -> Tuple[float, float]:
        """Get width and depth multipliers from model name."""
        config_map = {
            YOLOv5FaceWeights.YOLOV5N: (0.25, 0.33),  # nano
            YOLOv5FaceWeights.YOLOV5S: (0.50, 0.33),  # small
            YOLOv5FaceWeights.YOLOV5M: (0.75, 0.67),  # medium
            YOLOv5FaceWeights.YOLOV5L: (1.00, 1.00),  # large
        }
        return config_map.get(model_name, (0.50, 0.33))

    def _init_grids(self) -> None:
        """Initialize anchor grids for each detection scale."""
        self.grids = []
        self.anchor_grids = []

        for i, stride in enumerate(self.strides):
            grid_h = self.input_size // stride
            grid_w = self.input_size // stride

            # Create grid
            yv, xv = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing='ij')
            grid = np.stack([xv, yv], axis=-1).astype(np.float32)
            grid = grid.reshape(1, grid_h, grid_w, 1, 2)
            self.grids.append(grid)

            # Create anchor grid
            anchors = self.anchors[i].reshape(1, 1, 1, 3, 2)
            self.anchor_grids.append(anchors)

    def preprocess(self, image: np.ndarray) -> Tuple[mx.array, float, Tuple[int, int]]:
        """
        Preprocess image for MLX inference.

        Args:
            image: Input image in BGR format.

        Returns:
            Tuple of (input_tensor, scale, padding).
        """
        img_h, img_w = image.shape[:2]

        # Calculate scale ratio
        scale = min(self.input_size / img_h, self.input_size / img_w)
        new_h, new_w = int(img_h * scale), int(img_w * scale)

        # Resize image
        img_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image (letterbox)
        img_padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)

        # Calculate padding
        pad_h = (self.input_size - new_h) // 2
        pad_w = (self.input_size - new_w) // 2

        # Place resized image in center
        img_padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = img_resized

        # Convert to RGB and normalize
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0

        # Add batch dimension (H, W, C) -> (1, H, W, C) for NHWC format
        img_batch = np.expand_dims(img_normalized, axis=0)

        return mx.array(img_batch), scale, (pad_w, pad_h)

    def inference(self, input_tensor: mx.array) -> List[mx.array]:
        """Perform MLX inference."""
        outputs = self.model(input_tensor)

        # Force computation
        for out in outputs:
            synchronize(out)

        return outputs

    def postprocess(
        self,
        outputs: List[np.ndarray],
        scale: float,
        padding: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process model outputs into detections.

        Args:
            outputs: Raw model outputs per scale.
            scale: Scale ratio used in preprocessing.
            padding: Padding used in preprocessing.

        Returns:
            Tuple of (detections, landmarks).
        """
        all_boxes = []
        all_scores = []
        all_landmarks = []

        for i, (output, grid, anchor_grid, stride) in enumerate(
            zip(outputs, self.grids, self.anchor_grids, self.strides)
        ):
            n, h, w, _ = output.shape
            num_anchors = 3
            num_outputs = 16

            # Reshape: (N, H, W, anchors * outputs) -> (N, H, W, anchors, outputs)
            output = output.reshape(n, h, w, num_anchors, num_outputs)

            # Apply sigmoid to xy, objectness, and class scores
            xy = 1 / (1 + np.exp(-output[..., :2]))  # sigmoid
            wh = output[..., 2:4]
            obj = 1 / (1 + np.exp(-output[..., 4:5]))
            cls = 1 / (1 + np.exp(-output[..., 5:6]))
            landmarks = output[..., 6:16]

            # Decode bounding boxes
            xy = (xy * 2 - 0.5 + grid) * stride
            wh = (np.exp(wh) * 2) ** 2 * anchor_grid

            # Decode landmarks (same as bbox)
            for k in range(5):
                landmarks[..., k * 2] = landmarks[..., k * 2] * anchor_grid[..., 0] + grid[..., 0] * stride
                landmarks[..., k * 2 + 1] = landmarks[..., k * 2 + 1] * anchor_grid[..., 1] + grid[..., 1] * stride

            # Compute confidence
            conf = obj * cls

            # Flatten spatial and anchor dimensions
            xy = xy.reshape(-1, 2)
            wh = wh.reshape(-1, 2)
            conf = conf.reshape(-1)
            landmarks = landmarks.reshape(-1, 10)

            # Filter by confidence
            mask = conf >= self.conf_thresh
            if not np.any(mask):
                continue

            xy = xy[mask]
            wh = wh[mask]
            conf = conf[mask]
            landmarks = landmarks[mask]

            # Convert xywh to xyxy
            x1y1 = xy - wh / 2
            x2y2 = xy + wh / 2
            boxes = np.concatenate([x1y1, x2y2], axis=-1)

            all_boxes.append(boxes)
            all_scores.append(conf)
            all_landmarks.append(landmarks)

        if not all_boxes:
            return np.array([]), np.array([])

        # Concatenate all scales
        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        landmarks = np.concatenate(all_landmarks, axis=0)

        # Apply NMS
        detections_for_nms = np.hstack([boxes, scores[:, None]]).astype(np.float32)
        keep = non_max_suppression(detections_for_nms, self.nms_thresh)

        if len(keep) == 0:
            return np.array([]), np.array([])

        # Limit to max_det
        keep = keep[: self.max_det]
        boxes = boxes[keep]
        scores = scores[keep]
        landmarks = landmarks[keep]

        # Scale back to original image coordinates
        pad_w, pad_h = padding
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale

        # Scale landmarks
        for k in range(5):
            landmarks[:, k * 2] = (landmarks[:, k * 2] - pad_w) / scale
            landmarks[:, k * 2 + 1] = (landmarks[:, k * 2 + 1] - pad_h) / scale

        # Reshape landmarks to (N, 5, 2)
        landmarks = landmarks.reshape(-1, 5, 2)

        # Combine boxes and scores
        detections = np.concatenate([boxes, scores[:, None]], axis=1)

        return detections, landmarks

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
            max_num: Maximum detections to return (0 = all).
            metric: Ranking metric ('default' or 'max').
            center_weight: Weight for center-based ranking.

        Returns:
            List of face dictionaries with 'bbox', 'confidence', 'landmarks'.
        """
        original_height, original_width = image.shape[:2]

        # Preprocess
        input_tensor, scale, padding = self.preprocess(image)

        # MLX inference
        mlx_outputs = self.inference(input_tensor)

        # Convert to numpy for postprocessing
        outputs = [to_numpy(out) for out in mlx_outputs]

        # Postprocess
        detections, landmarks = self.postprocess(outputs, scale, padding)

        # Handle no detections
        if len(detections) == 0:
            return []

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
