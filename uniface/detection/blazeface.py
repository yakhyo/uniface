# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from uniface.common import validate_image
from uniface.constants import BlazeFaceWeights
from uniface.log import Logger
from uniface.model_store import verify_model_weights
from uniface.onnx_utils import create_onnx_session
from uniface.types import Face

from .base import BaseDetector

__all__ = ['BlazeFace']

# Keypoint order: right eye, left eye, nose tip, mouth center, right ear, left ear.
# FaceMesh reads rows 0/1 for the ROI roll angle
_NUM_KEYPOINTS = 6

# SsdAnchorsCalculator config for face_detection_short_range; hardcoded because every
# output dimension except the input size is a dynamic symbol in the ONNX metadata
_ANCHOR_STRIDES = (8, 16, 16, 16)
_NUM_ANCHORS = 896

# MediaPipe's score_clipping_thresh, applied before the sigmoid
_SCORE_CLIP = 100.0


def _generate_face_anchors(input_size: int) -> np.ndarray:
    """Build the SSD anchor centers for the short-range face detector.

    Layers sharing a stride stack their anchors per cell, giving 16x16x2 + 8x8x6 = 896
    anchors at 128px (aspect ratio 1.0, one interpolated scale, fixed anchor size).

    Args:
        input_size: The model's input edge length in pixels.

    Returns:
        Normalized (896, 2) anchor centers.

    Raises:
        ValueError: If the anchor count does not match the exported head.
    """
    anchors = []
    idx = 0
    while idx < len(_ANCHOR_STRIDES):
        last = idx
        while last < len(_ANCHOR_STRIDES) and _ANCHOR_STRIDES[last] == _ANCHOR_STRIDES[idx]:
            last += 1
        repeats = 2 * (last - idx)
        cells = input_size // _ANCHOR_STRIDES[idx]
        for y in range(cells):
            for x in range(cells):
                anchors.extend([((x + 0.5) / cells, (y + 0.5) / cells)] * repeats)
        idx = last

    if len(anchors) != _NUM_ANCHORS:
        raise ValueError(f'Expected {_NUM_ANCHORS} anchors for a {input_size}px input, generated {len(anchors)}')

    return np.array(anchors)


def _weighted_nms(detections: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Apply MediaPipe's *weighted* non-maximum suppression.

    Overlapping candidates are score-averaged into the winner rather than discarded,
    boxes and keypoints alike, so `common.non_max_suppression` cannot be reused.

    Args:
        detections: Rows of (score, cx, cy, w, h, kp0x, kp0y, ...), normalized.
        iou_threshold: Overlap above which candidates are blended into the winner.

    Returns:
        Blended detections in the same row layout, highest score first.
    """
    remaining = detections[np.argsort(-detections[:, 0])]
    output = []
    while len(remaining):
        top = remaining[0]
        x1 = remaining[:, 1] - remaining[:, 3] / 2
        y1 = remaining[:, 2] - remaining[:, 4] / 2
        x2 = remaining[:, 1] + remaining[:, 3] / 2
        y2 = remaining[:, 2] + remaining[:, 4] / 2
        tx1, ty1 = top[1] - top[3] / 2, top[2] - top[4] / 2
        tx2, ty2 = top[1] + top[3] / 2, top[2] + top[4] / 2

        inter = np.maximum(0, np.minimum(x2, tx2) - np.maximum(x1, tx1)) * np.maximum(
            0, np.minimum(y2, ty2) - np.maximum(y1, ty1)
        )
        union = (x2 - x1) * (y2 - y1) + (tx2 - tx1) * (ty2 - ty1) - inter
        iou = inter / np.maximum(union, 1e-9)

        # The winner joins its own blend unconditionally. Relying on its self-IoU clearing
        # the threshold loops forever when it cannot: iou_threshold=1.0 fails the strict
        # `>`, and a zero-area box scores an IoU of 0 against itself.
        merge = iou > iou_threshold
        merge[0] = True

        overlapping = remaining[merge]
        weights = overlapping[:, :1]
        blended = top.copy()
        total_weight = weights.sum()
        if total_weight > 0:
            blended[1:] = (overlapping[:, 1:] * weights).sum(axis=0) / total_weight
        output.append(blended)
        remaining = remaining[~merge]

    return np.array(output)


class BlazeFace(BaseDetector):
    """Short-range face detector based on the BlazeFace architecture (MediaPipe).

    Title: "BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs"
    Paper: https://arxiv.org/abs/1907.05047
    Code: https://github.com/google-ai-edge/mediapipe

    An SSD detector over two anchor grids on a 128x128 letterboxed image, decoded with
    MediaPipe's *weighted* NMS: overlapping candidates are score-averaged, not discarded.
    MediaPipe runs this detector ahead of its face mesh model, so feeding `BlazeFace`
    boxes to `FaceMesh` reproduces MediaPipe's own output.

    Note:
        Emits 6 MediaPipe keypoints rather than the 5-point alignment template, so
        `supports_alignment` is False: recognition, quality scoring, and XSeg parsing
        cannot consume them, and `FaceAnalyzer` disables recognition with a warning.
        Short-range covers faces within roughly 2m and scores below SCRFD/YOLOv8 on
        WIDER FACE; choose it for its 460KB footprint or for MediaPipe parity.

    Args:
        model_name (BlazeFaceWeights): Predefined model enum. Defaults to BlazeFaceWeights.DEFAULT.
        confidence_threshold (float): Confidence threshold for filtering detections. Defaults to 0.5.
        nms_threshold (float): IoU above which candidates are blended. Defaults to 0.3.
        providers (list[str] | None): ONNX Runtime execution providers. If None, auto-detects
            the best available provider. Example: ['CPUExecutionProvider'] to force CPU.

    Attributes:
        model_name (BlazeFaceWeights): Selected model variant.
        confidence_threshold (float): Threshold used to filter low-confidence detections.
        nms_threshold (float): Threshold used during weighted NMS to blend overlapping boxes.
        input_size (int): Input edge length read from the model. 128 for the short-range weights.
        anchors (np.ndarray): Precomputed anchor centers with shape (896, 2), normalized.
        _model_path (str): Absolute path to the downloaded/verified model weights.

    Raises:
        ValueError: If the generated anchor count does not match the exported head.
        RuntimeError: If the ONNX model fails to load or initialize.
    """

    supports_landmarks = True
    supports_alignment = False

    def __init__(
        self,
        *,
        model_name: BlazeFaceWeights = BlazeFaceWeights.DEFAULT,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        providers: list[str] | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            providers=providers,
        )

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.providers = providers

        Logger.info(
            f'Initializing BlazeFace with model={self.model_name}, '
            f'confidence_threshold={self.confidence_threshold}, '
            f'nms_threshold={self.nms_threshold}'
        )

        self._model_path = verify_model_weights(self.model_name)
        Logger.info(f'Verified model weights located at: {self._model_path}')

        self._initialize_model(self._model_path)

    def _initialize_model(self, model_path: str) -> None:
        """Initialize the ONNX session and precompute the SSD anchors.

        Args:
            model_path: The file path to the ONNX model.

        Raises:
            RuntimeError: If the model fails to load.
        """
        try:
            self.session = create_onnx_session(model_path, providers=self.providers)
            self.input_names = self.session.get_inputs()[0].name
            self.output_names = [x.name for x in self.session.get_outputs()]

            # Only the input size survives export as a concrete dimension
            self.input_size = int(self.session.get_inputs()[0].shape[2])
            self.anchors = _generate_face_anchors(self.input_size)

            Logger.info(f'Successfully initialized the model from {model_path}')
        except Exception as e:
            Logger.error(f"Failed to load model from '{model_path}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize model session for '{model_path}'") from e

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Convert a letterboxed BGR canvas into the network input tensor.

        BlazeFace expects RGB normalized to [-1, 1], unlike the landmark models' [0, 1].

        Args:
            image: Letterboxed canvas with shape (input_size, input_size, 3) in BGR.

        Returns:
            Preprocessed tensor with shape (1, 3, input_size, input_size).
        """
        rgb = image[:, :, ::-1].astype(np.float32)
        blob = np.transpose((rgb - 127.5) / 127.5, (2, 0, 1))
        return blob[np.newaxis]

    def postprocess(
        self,
        outputs: list[np.ndarray],
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode anchor predictions into full-image detections.

        Args:
            outputs: Raw outputs [regressors (1, 896, 16), logits (1, 896, 1)].
            scale: Letterbox scale factor applied in `detect`.
            pad_x: Horizontal letterbox padding in canvas pixels.
            pad_y: Vertical letterbox padding in canvas pixels.

        Returns:
            Tuple of (bboxes (N, 4), keypoints (N, 6, 2), scores (N,)) in
            full-image pixel coordinates.
        """
        regressors, logits = outputs
        size = self.input_size

        # Clipped before the sigmoid: strongly-negative anchor logits overflow np.exp
        raw = np.clip(logits[0].ravel().astype(np.float64), -_SCORE_CLIP, _SCORE_CLIP)
        scores = 1.0 / (1.0 + np.exp(-raw))
        keep = scores >= self.confidence_threshold
        if not keep.any():
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0, _NUM_KEYPOINTS, 2), dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )

        reg, anchor = regressors[0][keep].astype(np.float64), self.anchors[keep]
        rows = np.empty((int(keep.sum()), 5 + 2 * _NUM_KEYPOINTS))
        rows[:, 0] = scores[keep]
        rows[:, 1:3] = reg[:, 0:2] / size + anchor  # center x, y
        rows[:, 3:5] = reg[:, 2:4] / size  # width, height
        for k in range(_NUM_KEYPOINTS):
            rows[:, 5 + 2 * k : 7 + 2 * k] = reg[:, 4 + 2 * k : 6 + 2 * k] / size + anchor

        rows = _weighted_nms(rows, self.nms_threshold)

        def unletterbox(xy_normalized: np.ndarray) -> np.ndarray:
            return (xy_normalized * size - (pad_x, pad_y)) / scale

        centers = unletterbox(rows[:, 1:3])
        halves = rows[:, 3:5] * size / scale / 2.0
        bboxes = np.concatenate([centers - halves, centers + halves], axis=1)
        keypoints = unletterbox(rows[:, 5:].reshape(-1, _NUM_KEYPOINTS, 2))

        return bboxes.astype(np.float32), keypoints.astype(np.float32), rows[:, 0].astype(np.float32)

    def detect(
        self,
        image: np.ndarray,
        *,
        max_num: int = 0,
        metric: Literal['default', 'max'] = 'max',
        center_weight: float = 2.0,
    ) -> list[Face]:
        """Detect faces in an image.

        Args:
            image (np.ndarray): Input image as a NumPy array of shape (H, W, C) in BGR.
            max_num (int): Maximum number of detections to return. Use 0 for all.
            metric (Literal["default", "max"]): Metric for ranking when `max_num` is set.
                - "default": Prioritize detections closer to the image center.
                - "max": Prioritize detections with larger bounding box areas.
            center_weight (float): Penalty weight for distance from center under
                the "default" metric. Defaults to 2.0.

        Returns:
            list[Face]: List of Face objects, each containing:
                - bbox (np.ndarray): Bounding box with shape (4,) as [x1, y1, x2, y2]
                - confidence (float): Detection confidence score (0.0 to 1.0)
                - landmarks (np.ndarray): 6-point MediaPipe keypoints, shape (6, 2)

        Raises:
            ValueError: If the image is empty, not 3-channel BGR, or not uint8.

        Example:
            >>> faces = detector.detect(image)
            >>> for face in faces:
            ...     bbox = face.bbox  # np.ndarray with shape (4,)
            ...     keypoints = face.landmarks  # np.ndarray with shape (6, 2)
        """
        # This detector warps the input itself, so it never reaches the resize helpers
        # that validate for the other detectors.
        validate_image(image)

        height, width = image.shape[:2]
        size = self.input_size

        # Aspect-preserving letterbox onto a square canvas, centered.
        scale = size / max(height, width)
        pad_x, pad_y = (size - width * scale) / 2.0, (size - height * scale) / 2.0
        matrix = np.array([[scale, 0.0, pad_x], [0.0, scale, pad_y]])
        canvas = cv2.warpAffine(image, matrix, (size, size))

        image_tensor = self.preprocess(canvas)
        outputs = self.session.run(self.output_names, {self.input_names: image_tensor})
        bboxes, keypoints, scores = self.postprocess(outputs, scale, pad_x, pad_y)

        if bboxes.shape[0] == 0:
            return []

        detections = np.hstack((bboxes, scores[:, None])).astype(np.float32, copy=False)

        detections, keypoints = self._select_top_detections(
            detections, keypoints, max_num, (height, width), metric, center_weight
        )

        return self._detections_to_faces(detections, keypoints)
