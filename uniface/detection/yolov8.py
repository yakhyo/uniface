# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""
YOLOv8-Face detector implementation.

Uses anchor-free design with DFL (Distribution Focal Loss) for bbox regression.
Reference: https://github.com/yakhyo/yolov8-face-onnx-inference
"""

from typing import Any, Literal

import cv2
import numpy as np

from uniface.common import non_max_suppression
from uniface.constants import YOLOv8FaceWeights
from uniface.log import Logger
from uniface.model_store import verify_model_weights
from uniface.onnx_utils import create_onnx_session
from uniface.types import Face

from .base import BaseDetector

# Optional torchvision import for faster NMS
try:
    import torch
    import torchvision

    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

__all__ = ['YOLOv8Face']


class YOLOv8Face(BaseDetector):
    """
    Face detector based on the YOLOv8-Face architecture.

    Uses anchor-free design with DFL (Distribution Focal Loss) for bbox regression.
    Outputs 3 feature maps at different scales for multi-scale detection.

    Reference: https://github.com/yakhyo/yolov8-face-onnx-inference

    Args:
        model_name (YOLOv8FaceWeights): Predefined model enum (e.g., `YOLOV8N`).
            Specifies the YOLOv8-Face variant to load. Defaults to YOLOV8N.
        confidence_threshold (float): Confidence threshold for filtering detections. Defaults to 0.5.
        nms_threshold (float): Non-Maximum Suppression threshold. Defaults to 0.45.
        input_size (int): Input image size. Defaults to 640.
            Note: ONNX model is fixed at 640. Changing this will cause inference errors.
        nms_mode (str): NMS calculation method. Options: 'torchvision' (faster, requires torch)
            or 'numpy' (no dependencies). Defaults to 'numpy'.
        providers (list[str] | None): ONNX Runtime execution providers. If None, auto-detects
            the best available provider. Example: ['CPUExecutionProvider'] to force CPU.
        **kwargs: Advanced options:
            max_det (int): Maximum number of detections to return. Defaults to 750.

    Attributes:
        model_name (YOLOv8FaceWeights): Selected model variant.
        confidence_threshold (float): Threshold used to filter low-confidence detections.
        nms_threshold (float): Threshold used during NMS to suppress overlapping boxes.
        input_size (int): Image size to which inputs are resized before inference.
        nms_mode (str): NMS calculation method being used.
        max_det (int): Maximum number of detections to return.
        _model_path (str): Absolute path to the downloaded/verified model weights.

    Raises:
        ValueError: If the model weights are invalid or not found.
        RuntimeError: If the ONNX model fails to load or initialize.
    """

    def __init__(
        self,
        *,
        model_name: YOLOv8FaceWeights = YOLOv8FaceWeights.YOLOV8N,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        input_size: int = 640,
        nms_mode: Literal['torchvision', 'numpy'] = 'numpy',
        providers: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            input_size=input_size,
            nms_mode=nms_mode,
            providers=providers,
            **kwargs,
        )
        self._supports_landmarks = True  # YOLOv8-Face supports landmarks

        # Validate input size
        if input_size != 640:
            raise ValueError(
                f'YOLOv8Face only supports input_size=640 (got {input_size}). The ONNX model has a fixed input shape.'
            )

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.providers = providers

        # Set NMS mode with automatic fallback
        if nms_mode == 'torchvision' and not TORCHVISION_AVAILABLE:
            Logger.warning('torchvision not available, falling back to numpy NMS')
            self.nms_mode = 'numpy'
        else:
            self.nms_mode = nms_mode

        # Advanced options from kwargs
        self.max_det = kwargs.get('max_det', 750)

        # YOLOv8 strides for 640x640 input (3 feature maps: 80x80, 40x40, 20x20)
        self.strides = [8, 16, 32]

        Logger.info(
            f'Initializing YOLOv8Face with model={self.model_name}, confidence_threshold={self.confidence_threshold}, '
            f'nms_threshold={self.nms_threshold}, input_size={self.input_size}, nms_mode={self.nms_mode}'
        )

        # Get path to model weights
        self._model_path = verify_model_weights(self.model_name)
        Logger.info(f'Verified model weights located at: {self._model_path}')

        # Initialize model
        self._initialize_model(self._model_path)

    def _initialize_model(self, model_path: str) -> None:
        """
        Initializes an ONNX model session from the given path.

        Args:
            model_path (str): The file path to the ONNX model.

        Raises:
            RuntimeError: If the model fails to load, logs an error and raises an exception.
        """
        try:
            self.session = create_onnx_session(model_path, providers=self.providers)
            self.input_names = self.session.get_inputs()[0].name
            self.output_names = [x.name for x in self.session.get_outputs()]
            Logger.info(f'Successfully initialized the model from {model_path}')
        except Exception as e:
            Logger.error(f"Failed to load model from '{model_path}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize model session for '{model_path}'") from e

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """
        Preprocess image for inference (letterbox resize with center padding).

        Args:
            image (np.ndarray): Input image (BGR format)

        Returns:
            Tuple[np.ndarray, float, Tuple[int, int]]: Preprocessed image, scale ratio, and padding (pad_w, pad_h)
        """
        # Get original image shape
        img_h, img_w = image.shape[:2]

        # Calculate scale ratio
        scale = min(self.input_size / img_h, self.input_size / img_w)
        new_h, new_w = int(img_h * scale), int(img_w * scale)

        # Resize image
        img_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image with gray background (114, 114, 114)
        img_padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)

        # Calculate padding (center the image)
        pad_h = (self.input_size - new_h) // 2
        pad_w = (self.input_size - new_w) // 2

        # Place resized image in center
        img_padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = img_resized

        # Convert BGR to RGB and normalize
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0

        # Transpose to CHW format (HWC -> CHW) and add batch dimension
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)
        img_batch = np.ascontiguousarray(img_batch)

        return img_batch, scale, (pad_w, pad_h)

    def inference(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        """Perform model inference on the preprocessed image tensor.

        Args:
            input_tensor (np.ndarray): Preprocessed input tensor.

        Returns:
            List[np.ndarray]: Raw model outputs (3 feature maps).
        """
        return self.session.run(self.output_names, {self.input_names: input_tensor})

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax values for array x along specified axis."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def postprocess(
        self,
        predictions: list[np.ndarray],
        scale: float,
        padding: tuple[int, int],
        original_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Postprocess model predictions with DFL decoding and coordinate scaling.

        Args:
            predictions (List[np.ndarray]): Raw model outputs (3 feature maps)
            scale (float): Scale ratio used in preprocessing
            padding (Tuple[int, int]): Padding (pad_w, pad_h) used in preprocessing
            original_shape (Tuple[int, int]): Original image shape (height, width)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Filtered detections and landmarks
                - detections: [N, 5] as [x1, y1, x2, y2, conf]
                - landmarks: [N, 5, 2] for each detection
        """
        # YOLOv8-Face outputs 3 feature maps with Pose head
        # Each output: (1, 80, H, W) where 80 = 64 (bbox DFL) + 1 (class) + 15 (5 keypoints * 3)

        boxes_list = []
        scores_list = []
        landmarks_list = []

        for pred, stride in zip(predictions, self.strides, strict=False):
            # pred shape: (1, 80, H, W)
            batch_size, channels, height, width = pred.shape

            # Reshape: (1, 80, H, W) -> (1, 80, H*W) -> (1, H*W, 80) -> (H*W, 80)
            pred = pred.reshape(batch_size, channels, -1).transpose(0, 2, 1)[0]

            # Create grid with 0.5 offset (matching PyTorch's make_anchors)
            grid_y, grid_x = np.meshgrid(np.arange(height) + 0.5, np.arange(width) + 0.5, indexing='ij')
            grid_x = grid_x.flatten()
            grid_y = grid_y.flatten()

            # Extract components
            bbox_pred = pred[:, :64]  # DFL bbox prediction (64 channels = 4 * 16)
            cls_conf = pred[:, 64]  # Class confidence (1 channel)
            kpt_pred = pred[:, 65:]  # Keypoints (15 channels = 5 points * 3: x, y, visibility)

            # Decode bounding boxes from DFL
            bbox_pred = bbox_pred.reshape(-1, 4, 16)
            bbox_dist = self._softmax(bbox_pred, axis=-1) @ np.arange(16)

            # Convert distances to xyxy format
            x1 = (grid_x - bbox_dist[:, 0]) * stride
            y1 = (grid_y - bbox_dist[:, 1]) * stride
            x2 = (grid_x + bbox_dist[:, 2]) * stride
            y2 = (grid_y + bbox_dist[:, 3]) * stride
            boxes = np.stack([x1, y1, x2, y2], axis=-1)

            # Decode keypoints: kpt = (kpt * 2.0 + grid) * stride
            kpt_grid_y, kpt_grid_x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
            kpt_grid_x = kpt_grid_x.flatten()
            kpt_grid_y = kpt_grid_y.flatten()

            kpt_pred = kpt_pred.reshape(-1, 5, 3)  # 5 points * (x, y, visibility)
            kpt_x = (kpt_pred[:, :, 0] * 2.0 + kpt_grid_x[:, None]) * stride
            kpt_y = (kpt_pred[:, :, 1] * 2.0 + kpt_grid_y[:, None]) * stride
            # Ignore visibility (kpt_pred[:, :, 2]) for uniface compatibility
            landmarks = np.stack([kpt_x, kpt_y], axis=-1).reshape(-1, 10)

            # Apply sigmoid to class confidence
            scores = 1 / (1 + np.exp(-cls_conf))

            boxes_list.append(boxes)
            scores_list.append(scores)
            landmarks_list.append(landmarks)

        # Concatenate all predictions from all feature maps
        boxes = np.concatenate(boxes_list, axis=0)
        scores = np.concatenate(scores_list, axis=0)
        landmarks = np.concatenate(landmarks_list, axis=0)

        # Filter by confidence threshold
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        landmarks = landmarks[mask]

        if len(boxes) == 0:
            return np.array([]), np.array([])

        # Apply NMS based on selected mode
        if self.nms_mode == 'torchvision':
            keep = torchvision.ops.nms(
                torch.tensor(boxes, dtype=torch.float32),
                torch.tensor(scores, dtype=torch.float32),
                self.nms_threshold,
            ).numpy()
        else:
            detections_for_nms = np.hstack((boxes, scores[:, None])).astype(np.float32, copy=False)
            keep = non_max_suppression(detections_for_nms, self.nms_threshold)

        if len(keep) == 0:
            return np.array([]), np.array([])

        # Limit to max_det
        keep = keep[: self.max_det]
        boxes = boxes[keep]
        scores = scores[keep]
        landmarks = landmarks[keep]

        # === SCALE TO ORIGINAL IMAGE COORDINATES ===
        pad_w, pad_h = padding

        # Scale boxes back to original image coordinates
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale

        # Clip boxes to image boundaries
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_shape[1])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_shape[0])

        # Scale landmarks back to original image coordinates
        landmarks[:, 0::2] = (landmarks[:, 0::2] - pad_w) / scale  # x coordinates
        landmarks[:, 1::2] = (landmarks[:, 1::2] - pad_h) / scale  # y coordinates

        # Reshape landmarks to (N, 5, 2)
        landmarks = landmarks.reshape(-1, 5, 2)

        # Combine box and score
        detections = np.concatenate([boxes, scores[:, None]], axis=1)

        return detections, landmarks

    def detect(
        self,
        image: np.ndarray,
        *,
        max_num: int = 0,
        metric: Literal['default', 'max'] = 'max',
        center_weight: float = 2.0,
    ) -> list[Face]:
        """
        Perform face detection on an input image and return bounding boxes and facial landmarks.

        Args:
            image (np.ndarray): Input image as a NumPy array of shape (H, W, C) in BGR format.
            max_num (int): Maximum number of detections to return. Use 0 to return all detections. Defaults to 0.
            metric (Literal["default", "max"]): Metric for ranking detections when `max_num` is limited.
                - "default": Prioritize detections closer to the image center.
                - "max": Prioritize detections with larger bounding box areas.
            center_weight (float): Weight for penalizing detections farther from the image center
                when using the "default" metric. Defaults to 2.0.

        Returns:
            List[Face]: List of Face objects, each containing:
                - bbox (np.ndarray): Bounding box coordinates with shape (4,) as [x1, y1, x2, y2]
                - confidence (float): Detection confidence score (0.0 to 1.0)
                - landmarks (np.ndarray): 5-point facial landmarks with shape (5, 2)

        Example:
            >>> faces = detector.detect(image)
            >>> for face in faces:
            ...     bbox = face.bbox  # np.ndarray with shape (4,)
            ...     confidence = face.confidence  # float
            ...     landmarks = face.landmarks  # np.ndarray with shape (5, 2)
        """
        original_height, original_width = image.shape[:2]

        # Preprocess
        image_tensor, scale, padding = self.preprocess(image)

        # ONNXRuntime inference
        outputs = self.inference(image_tensor)

        # Postprocess with original image shape for clipping
        detections, landmarks = self.postprocess(outputs, scale, padding, (original_height, original_width))

        # Handle case when no faces are detected
        if len(detections) == 0:
            return []

        if 0 < max_num < detections.shape[0]:
            # Calculate area of detections
            area = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])

            # Calculate offsets from image center
            center = (original_height // 2, original_width // 2)
            offsets = np.vstack(
                [
                    (detections[:, 0] + detections[:, 2]) / 2 - center[1],
                    (detections[:, 1] + detections[:, 3]) / 2 - center[0],
                ]
            )

            # Calculate scores based on the chosen metric
            offset_dist_squared = np.sum(np.power(offsets, 2.0), axis=0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * center_weight

            # Sort by scores and select top `max_num`
            sorted_indices = np.argsort(values)[::-1][:max_num]
            detections = detections[sorted_indices]
            landmarks = landmarks[sorted_indices]

        faces = []
        for i in range(detections.shape[0]):
            face = Face(
                bbox=detections[i, :4],
                confidence=float(detections[i, 4]),
                landmarks=landmarks[i],
            )
            faces.append(face)

        return faces
