# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from uniface.common import non_max_suppression, validate_image
from uniface.constants import CenterFaceWeights
from uniface.log import Logger
from uniface.model_store import verify_model_weights
from uniface.onnx_utils import create_onnx_session
from uniface.types import Face

from .base import BaseDetector

__all__ = ['CenterFace']

# CenterFace predicts on a single feature map downsampled 4x from the input
_STRIDE = 4

# The FPN needs both input sides to be multiples of 32
_SIZE_DIVISOR = 32


class CenterFace(BaseDetector):
    """Anchor-free face detector based on the CenterFace architecture.

    Title: "CenterFace: Joint Face Detection and Alignment Using Face as Point"
    Paper: https://arxiv.org/abs/1911.03599
    Code: https://github.com/Star-Clouds/CenterFace

    Faces are detected as center points on a single stride-4 feature map with four
    prediction heads: center heatmap, box scale, center offset, and 5-point landmarks.

    Note:
        Landmarks are decoded from a single coarse feature-map cell per face and are less
        precise than SCRFD/RetinaFace; accuracy also degrades faster for in-plane rotated
        faces (beyond ~20-30 degrees), upstream included. Best suited for upright faces.

    Args:
        model_name (CenterFaceWeights): Predefined model enum. Defaults to CenterFaceWeights.DEFAULT.
        confidence_threshold (float): Confidence threshold for filtering detections. Defaults to 0.35.
        nms_threshold (float): Non-Maximum Suppression threshold. Defaults to 0.3.
        input_size (tuple[int, int] | None): Upper bound on the inference resolution as
            (width, height), not a fixed shape: larger images are scaled down to fit and
            smaller ones are left alone, preserving aspect ratio. Defaults to (640, 640).
            Pass None to always run at native resolution, like upstream.
        providers (list[str] | None): ONNX Runtime execution providers. If None, auto-detects
            the best available provider. Example: ['CPUExecutionProvider'] to force CPU.

    Attributes:
        model_name (CenterFaceWeights): Selected model variant.
        confidence_threshold (float): Threshold used to filter low-confidence detections.
        nms_threshold (float): Threshold used during NMS to suppress overlapping boxes.
        input_size (tuple[int, int] | None): Maximum inference resolution, or None for native.
        _model_path (str): Absolute path to the downloaded/verified model weights.

    Raises:
        ValueError: If `input_size` is not strictly positive, or the model weights are invalid.
        RuntimeError: If the ONNX model fails to load or initialize.
    """

    supports_landmarks = True
    supports_alignment = True

    def __init__(
        self,
        *,
        model_name: CenterFaceWeights = CenterFaceWeights.DEFAULT,
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.3,
        input_size: tuple[int, int] | None = (640, 640),
        providers: list[str] | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            input_size=input_size,
            providers=providers,
        )
        if input_size is not None and (input_size[0] <= 0 or input_size[1] <= 0):
            raise ValueError(f'input_size must be strictly positive, got {input_size}')

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.providers = providers

        Logger.info(
            f'Initializing CenterFace with model={self.model_name}, '
            f'confidence_threshold={self.confidence_threshold}, '
            f'nms_threshold={self.nms_threshold}, input_size={self.input_size}'
        )

        self._model_path = verify_model_weights(self.model_name)
        Logger.info(f'Verified model weights located at: {self._model_path}')

        self._initialize_model(self._model_path)

    def _initialize_model(self, model_path: str) -> None:
        """Initialize an ONNX model session from the given path.

        Args:
            model_path: The file path to the ONNX model.

        Raises:
            RuntimeError: If the model fails to load.
        """
        try:
            self.session = create_onnx_session(model_path, providers=self.providers)
            self.input_names = self.session.get_inputs()[0].name
            self.output_names = [x.name for x in self.session.get_outputs()]
            Logger.info(f'Successfully initialized the model from {model_path}')
        except Exception as e:
            Logger.error(f"Failed to load model from '{model_path}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize model session for '{model_path}'") from e

    def _resize(self, image: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Resize an image to a network-compatible shape, preserving aspect ratio.

        Args:
            image: Input image with shape (H, W, C).

        Returns:
            A tuple of (resized image, width scale factor, height scale factor), where each
            scale factor maps original coordinates to resized ones.
        """
        height, width = image.shape[:2]

        ratio = 1.0
        if self.input_size is not None:
            max_width, max_height = self.input_size
            ratio = min(1.0, max_width / width, max_height / height)

        # Round up independently per axis, so the two scale factors need not match
        new_width = max(_SIZE_DIVISOR, int(np.ceil(width * ratio / _SIZE_DIVISOR) * _SIZE_DIVISOR))
        new_height = max(_SIZE_DIVISOR, int(np.ceil(height * ratio / _SIZE_DIVISOR) * _SIZE_DIVISOR))

        resized = cv2.resize(image, (new_width, new_height))

        return resized, new_width / width, new_height / height

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference.

        CenterFace consumes raw RGB pixel values (no mean subtraction or scaling).

        Args:
            image: Input image with shape (H, W, C) in BGR order.

        Returns:
            Preprocessed image tensor with shape (1, C, H, W).
        """
        image = image[:, :, ::-1].astype(np.float32)  # BGR to RGB
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = np.expand_dims(image, axis=0)

        return image

    def inference(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        """Perform model inference on the preprocessed image tensor.

        Args:
            input_tensor: Preprocessed input tensor with shape (1, C, H, W).

        Returns:
            List of raw model outputs: [heatmap, scale, offset, landmarks].
        """
        return self.session.run(self.output_names, {self.input_names: input_tensor})

    def postprocess(
        self,
        outputs: list[np.ndarray],
        image_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode center-point predictions into detection results.

        Args:
            outputs: Raw outputs [heatmap (1,1,H/4,W/4), scale (1,2,H/4,W/4),
                offset (1,2,H/4,W/4), landmarks (1,10,H/4,W/4)].
            image_size: Size of the model input image as (height, width).

        Returns:
            Tuple of (bboxes (N, 4), scores (N,), landmarks (N, 5, 2)).
        """
        heatmap, scale, offset, lms = outputs
        height, width = image_size

        heatmap = heatmap[0, 0]
        cy, cx = np.where(heatmap > self.confidence_threshold)
        if len(cy) == 0:
            empty = np.empty((0,), dtype=np.float32)
            return np.empty((0, 4), dtype=np.float32), empty, np.empty((0, 5, 2), dtype=np.float32)

        scores = heatmap[cy, cx]
        box_h = np.exp(scale[0, 0, cy, cx]) * _STRIDE
        box_w = np.exp(scale[0, 1, cy, cx]) * _STRIDE
        offset_y = offset[0, 0, cy, cx]
        offset_x = offset[0, 1, cy, cx]

        x1 = np.clip((cx + offset_x + 0.5) * _STRIDE - box_w / 2, 0, width)
        y1 = np.clip((cy + offset_y + 0.5) * _STRIDE - box_h / 2, 0, height)
        x2 = np.clip(x1 + box_w, 0, width)
        y2 = np.clip(y1 + box_h, 0, height)
        bboxes = np.stack([x1, y1, x2, y2], axis=1)

        # Landmarks are predicted relative to the box: (dy, dx) pairs per point
        points = lms[0][:, cy, cx]  # (10, N)
        kps_x = points[1::2].T * box_w[:, None] + x1[:, None]
        kps_y = points[0::2].T * box_h[:, None] + y1[:, None]
        landmarks = np.stack([kps_x, kps_y], axis=2)  # (N, 5, 2)

        return bboxes, scores, landmarks

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
            image (np.ndarray): Input image as a NumPy array of shape (H, W, C).
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

        image, scale_w, scale_h = self._resize(image)

        image_tensor = self.preprocess(image)

        outputs = self.inference(image_tensor)

        # Postprocessing
        bboxes, scores, landmarks = self.postprocess(outputs, image_size=image.shape[:2])

        # Handle case when no faces are detected
        if bboxes.shape[0] == 0:
            return []

        # Rounding each side up to a multiple of 32 skews the axes independently
        bboxes[:, 0::2] /= scale_w
        bboxes[:, 1::2] /= scale_h
        landmarks[..., 0] /= scale_w
        landmarks[..., 1] /= scale_h

        order = scores.argsort()[::-1]
        pre_det = np.hstack((bboxes, scores[:, None])).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        landmarks = landmarks[order]

        keep = non_max_suppression(pre_det, threshold=self.nms_threshold)

        detections = pre_det[keep, :]
        landmarks = landmarks[keep].astype(np.float32)

        # Filter to top max_num faces if requested
        detections, landmarks = self._select_top_detections(
            detections, landmarks, max_num, (original_height, original_width), metric, center_weight
        )

        return self._detections_to_faces(detections, landmarks)
