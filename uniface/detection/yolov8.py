# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from typing import Literal

import numpy as np

from uniface.common import letterbox_resize, non_max_suppression, softmax, validate_image
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
    """Face detector based on the YOLOv8-Face architecture.

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

    supports_landmarks = True
    supports_alignment = True

    def __init__(
        self,
        *,
        model_name: YOLOv8FaceWeights = YOLOv8FaceWeights.YOLOV8N,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        input_size: int = 640,
        nms_mode: Literal['torchvision', 'numpy'] = 'numpy',
        providers: list[str] | None = None,
        max_det: int = 750,
    ) -> None:
        super().__init__(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            input_size=input_size,
            nms_mode=nms_mode,
            providers=providers,
            max_det=max_det,
        )

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

        # Advanced options
        self.max_det = max_det

        # YOLOv8 strides for 640x640 input (3 feature maps: 80x80, 40x40, 20x20)
        self.strides = [8, 16, 32]
        self._dfl_proj = np.arange(16, dtype=np.float32)
        self._grid_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

        Logger.info(
            f'Initializing YOLOv8Face with model={self.model_name}, confidence_threshold={self.confidence_threshold}, '
            f'nms_threshold={self.nms_threshold}, input_size={self.input_size}, nms_mode={self.nms_mode}'
        )

        self._model_path = verify_model_weights(self.model_name)
        Logger.info(f'Verified model weights located at: {self._model_path}')

        self._initialize_model(self._model_path)

    def _initialize_model(self, model_path: str) -> None:
        """Initializes an ONNX model session from the given path.

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
        """Preprocess image using letterbox resize.

        Args:
            image: Input image in BGR format.

        Returns:
            Tuple of (preprocessed_tensor, scale_ratio, padding).
        """
        return letterbox_resize(image, self.input_size)

    def inference(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        """Perform model inference on the preprocessed image tensor.

        Args:
            input_tensor (np.ndarray): Preprocessed input tensor.

        Returns:
            list[np.ndarray]: Raw model outputs (3 feature maps).
        """
        return self.session.run(self.output_names, {self.input_names: input_tensor})

    def postprocess(
        self,
        predictions: list[np.ndarray],
        scale: float,
        padding: tuple[int, int],
        original_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Postprocess model predictions with DFL decoding and coordinate scaling.

        Args:
            predictions (list[np.ndarray]): Raw model outputs (3 feature maps)
            scale (float): Scale ratio used in preprocessing
            padding (tuple[int, int]): Padding (pad_w, pad_h) used in preprocessing
            original_shape (tuple[int, int]): Original image shape (height, width)

        Returns:
            tuple[np.ndarray, np.ndarray]: Filtered detections and landmarks
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
            _, channels, height, width = pred.shape

            # Reshape: (1, 80, H, W) -> (H*W, 80)
            pred_flat = pred[0].reshape(channels, -1).T

            # Clip logits before sigmoid to prevent float overflow on strongly negative values
            cls_conf = np.clip(pred_flat[:, 64], -50.0, 50.0)
            scores = (1.0 / (1.0 + np.exp(-cls_conf))).astype(np.float32, copy=False)

            # Filter by confidence before running DFL softmax and keypoint decoding
            mask = scores >= self.confidence_threshold
            if not np.any(mask):
                continue

            cache_key = (height, width)
            if cache_key in self._grid_cache:
                grid_x, grid_y = self._grid_cache[cache_key]
            else:
                gy, gx = np.meshgrid(
                    np.arange(height, dtype=np.float32),
                    np.arange(width, dtype=np.float32),
                    indexing='ij',
                )
                grid_x, grid_y = gx.ravel(), gy.ravel()
                if len(self._grid_cache) < 32:
                    self._grid_cache[cache_key] = (grid_x, grid_y)

            gx_m = grid_x[mask]
            gy_m = grid_y[mask]
            pred_m = pred_flat[mask]
            scores_m = scores[mask]

            # Decode bounding boxes from DFL (64 channels = 4 * 16)
            bbox_pred = pred_m[:, :64].reshape(-1, 4, 16)
            bbox_dist = softmax(bbox_pred, axis=-1) @ self._dfl_proj

            # Convert distances to xyxy format using 0.5 cell-center offset
            cx = gx_m + np.float32(0.5)
            cy = gy_m + np.float32(0.5)
            x1 = (cx - bbox_dist[:, 0]) * stride
            y1 = (cy - bbox_dist[:, 1]) * stride
            x2 = (cx + bbox_dist[:, 2]) * stride
            y2 = (cy + bbox_dist[:, 3]) * stride
            boxes = np.stack([x1, y1, x2, y2], axis=-1)

            # Decode keypoints: kpt = (kpt * 2.0 + grid) * stride (visibility channel ignored)
            kpt_pred = pred_m[:, 65:].reshape(-1, 5, 3)
            kpt_x = (kpt_pred[:, :, 0] * np.float32(2.0) + gx_m[:, None]) * stride
            kpt_y = (kpt_pred[:, :, 1] * np.float32(2.0) + gy_m[:, None]) * stride
            landmarks = np.stack([kpt_x, kpt_y], axis=-1).reshape(-1, 10)

            boxes_list.append(boxes)
            scores_list.append(scores_m)
            landmarks_list.append(landmarks)

        if not boxes_list:
            return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)

        # Concatenate filtered predictions from all feature maps
        boxes = np.concatenate(boxes_list, axis=0)
        scores = np.concatenate(scores_list, axis=0)
        landmarks = np.concatenate(landmarks_list, axis=0)

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
            return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)

        # Filter detections and limit to max_det
        keep = keep[: self.max_det]
        boxes = boxes[keep]
        scores = scores[keep]
        landmarks = landmarks[keep]

        # Scale back to original image coordinates
        pad_w, pad_h = padding
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale

        # Clip boxes to image boundaries
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_shape[1])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_shape[0])

        # Scale landmarks back to original image coordinates
        landmarks[:, 0::2] = (landmarks[:, 0::2] - pad_w) / scale  # x coordinates
        landmarks[:, 1::2] = (landmarks[:, 1::2] - pad_h) / scale  # y coordinates

        # Reshape landmarks to (N, 5, 2)
        landmarks = landmarks.reshape(-1, 5, 2).astype(np.float32, copy=False)

        # Combine results
        detections = np.concatenate([boxes, scores[:, None]], axis=1).astype(np.float32, copy=False)

        return detections, landmarks

    def detect(
        self,
        image: np.ndarray,
        *,
        max_num: int = 0,
        metric: Literal['default', 'max'] = 'max',
        center_weight: float = 2.0,
    ) -> list[Face]:
        """Perform face detection on an input image and return bounding boxes and facial landmarks.

        Args:
            image (np.ndarray): Input image as a NumPy array of shape (H, W, C) in BGR format.
            max_num (int): Maximum number of detections to return. Use 0 to return all detections. Defaults to 0.
            metric (Literal["default", "max"]): Metric for ranking detections when `max_num` is limited.
                - "default": Prioritize detections closer to the image center.
                - "max": Prioritize detections with larger bounding box areas.
            center_weight (float): Weight for penalizing detections farther from the image center
                when using the "default" metric. Defaults to 2.0.

        Returns:
            list[Face]: List of Face objects, each containing:
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
        validate_image(image)
        original_height, original_width = image.shape[:2]

        image_tensor, scale, padding = self.preprocess(image)

        # ONNXRuntime inference
        outputs = self.inference(image_tensor)

        # Postprocess with original image shape for clipping
        detections, landmarks = self.postprocess(outputs, scale, padding, (original_height, original_width))

        # Handle case when no faces are detected
        if len(detections) == 0:
            return []

        # Filter to top max_num faces if requested
        detections, landmarks = self._select_top_detections(
            detections, landmarks, max_num, (original_height, original_width), metric, center_weight
        )

        return self._detections_to_faces(detections, landmarks)
