# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from typing import Any, Literal

import cv2
import numpy as np

from uniface.common import non_max_suppression
from uniface.constants import YOLOv5FaceWeights
from uniface.log import Logger
from uniface.model_store import verify_model_weights
from uniface.onnx_utils import create_onnx_session
from uniface.types import Face

from .base import BaseDetector

__all__ = ['YOLOv5Face']


class YOLOv5Face(BaseDetector):
    """
    Face detector based on the YOLOv5-Face architecture.

    Title: "YOLO5Face: Why Reinventing a Face Detector"
    Paper: https://arxiv.org/abs/2105.12931
    Code: https://github.com/yakhyo/yolov5-face-onnx-inference (ONNX inference implementation)

    Args:
        model_name (YOLOv5FaceWeights): Predefined model enum (e.g., `YOLOV5S`).
            Specifies the YOLOv5-Face variant to load. Defaults to YOLOV5S.
        confidence_threshold (float): Confidence threshold for filtering detections. Defaults to 0.6.
        nms_threshold (float): Non-Maximum Suppression threshold. Defaults to 0.5.
        input_size (int): Input image size. Defaults to 640.
            Note: ONNX model is fixed at 640. Changing this will cause inference errors.
        **kwargs: Advanced options:
            max_det (int): Maximum number of detections to return. Defaults to 750.

    Attributes:
        model_name (YOLOv5FaceWeights): Selected model variant.
        confidence_threshold (float): Threshold used to filter low-confidence detections.
        nms_threshold (float): Threshold used during NMS to suppress overlapping boxes.
        input_size (int): Image size to which inputs are resized before inference.
        max_det (int): Maximum number of detections to return.
        _model_path (str): Absolute path to the downloaded/verified model weights.

    Raises:
        ValueError: If the model weights are invalid or not found.
        RuntimeError: If the ONNX model fails to load or initialize.
    """

    def __init__(
        self,
        *,
        model_name: YOLOv5FaceWeights = YOLOv5FaceWeights.YOLOV5S,
        confidence_threshold: float = 0.6,
        nms_threshold: float = 0.5,
        input_size: int = 640,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            input_size=input_size,
            **kwargs,
        )
        self._supports_landmarks = True  # YOLOv5-Face supports landmarks

        # Validate input size
        if input_size != 640:
            raise ValueError(
                f'YOLOv5Face only supports input_size=640 (got {input_size}). The ONNX model has a fixed input shape.'
            )

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        # Advanced options from kwargs
        self.max_det = kwargs.get('max_det', 750)

        Logger.info(
            f'Initializing YOLOv5Face with model={self.model_name}, confidence_threshold={self.confidence_threshold}, '
            f'nms_threshold={self.nms_threshold}, input_size={self.input_size}'
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
            self.session = create_onnx_session(model_path)
            self.input_names = self.session.get_inputs()[0].name
            self.output_names = [x.name for x in self.session.get_outputs()]
            Logger.info(f'Successfully initialized the model from {model_path}')
        except Exception as e:
            Logger.error(f"Failed to load model from '{model_path}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize model session for '{model_path}'") from e

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """
        Preprocess image for inference.

        Args:
            image (np.ndarray): Input image (BGR format)

        Returns:
            Tuple[np.ndarray, float, Tuple[int, int]]: Preprocessed image, scale ratio, and padding
        """
        # Get original image shape
        img_h, img_w = image.shape[:2]

        # Calculate scale ratio
        scale = min(self.input_size / img_h, self.input_size / img_w)
        new_h, new_w = int(img_h * scale), int(img_w * scale)

        # Resize image
        img_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image
        img_padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)

        # Calculate padding
        pad_h = (self.input_size - new_h) // 2
        pad_w = (self.input_size - new_w) // 2

        # Place resized image in center
        img_padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = img_resized

        # Convert to RGB and normalize
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
            List[np.ndarray]: Raw model outputs.
        """
        return self.session.run(self.output_names, {self.input_names: input_tensor})

    def postprocess(
        self,
        predictions: np.ndarray,
        scale: float,
        padding: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Postprocess model predictions.

        Args:
            predictions (np.ndarray): Raw model output
            scale (float): Scale ratio used in preprocessing
            padding (Tuple[int, int]): Padding used in preprocessing

        Returns:
            Tuple[np.ndarray, np.ndarray]: Filtered detections and landmarks
                - detections: [x1, y1, x2, y2, conf]
                - landmarks: [5, 2] for each detection
        """
        # predictions shape: (1, 25200, 16)
        # 16 = [x, y, w, h, obj_conf, cls_conf, 10 landmarks (5 points * 2 coords)]

        predictions = predictions[0]  # Remove batch dimension

        # Filter by confidence
        mask = predictions[:, 4] >= self.confidence_threshold
        predictions = predictions[mask]

        if len(predictions) == 0:
            return np.array([]), np.array([])

        # Convert from xywh to xyxy
        boxes = self._xywh2xyxy(predictions[:, :4])

        # Get confidence scores
        scores = predictions[:, 4]

        # Get landmarks (5 points, 10 coordinates)
        landmarks = predictions[:, 5:15].copy()

        # Apply NMS
        detections_for_nms = np.hstack((boxes, scores[:, None])).astype(np.float32, copy=False)
        keep = non_max_suppression(detections_for_nms, self.nms_threshold)

        if len(keep) == 0:
            return np.array([]), np.array([])

        # Filter detections and limit to max_det
        keep = keep[: self.max_det]
        boxes = boxes[keep]
        scores = scores[keep]
        landmarks = landmarks[keep]

        # Scale back to original image coordinates
        pad_w, pad_h = padding
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale

        # Scale landmarks
        for i in range(5):
            landmarks[:, i * 2] = (landmarks[:, i * 2] - pad_w) / scale
            landmarks[:, i * 2 + 1] = (landmarks[:, i * 2 + 1] - pad_h) / scale

        # Reshape landmarks to (N, 5, 2)
        landmarks = landmarks.reshape(-1, 5, 2)

        # Combine results
        detections = np.concatenate([boxes, scores[:, None]], axis=1)

        return detections, landmarks

    def _xywh2xyxy(self, x: np.ndarray) -> np.ndarray:
        """
        Convert bounding box format from xywh to xyxy.

        Args:
            x (np.ndarray): Boxes in [x, y, w, h] format

        Returns:
            np.ndarray: Boxes in [x1, y1, x2, y2] format
        """
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
        return y

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
            image (np.ndarray): Input image as a NumPy array of shape (H, W, C).
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
            ...     # Can pass landmarks directly to recognition
            ...     embedding = recognizer.get_normalized_embedding(image, face.landmarks)
        """

        original_height, original_width = image.shape[:2]

        # Preprocess
        image_tensor, scale, padding = self.preprocess(image)

        # ONNXRuntime inference
        outputs = self.inference(image_tensor)

        # Postprocess
        detections, landmarks = self.postprocess(outputs[0], scale, padding)

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
