# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from uniface.common import distance2bbox, distance2kps, non_max_suppression, resize_image
from uniface.constants import SCRFDWeights
from uniface.log import Logger
from uniface.model_store import verify_model_weights
from uniface.onnx_utils import create_onnx_session
from uniface.types import Face

from .base import BaseDetector

__all__ = ['SCRFD']


class SCRFD(BaseDetector):
    """
    Face detector based on the SCRFD architecture.

    Title: "Sample and Computation Redistribution for Efficient Face Detection"
    Paper: https://arxiv.org/abs/2105.04714
    Code: https://github.com/insightface/insightface

    Args:
        model_name (SCRFDWeights): Predefined model enum (e.g., `SCRFD_10G_KPS`).
            Specifies the SCRFD variant to load. Defaults to SCRFD_10G_KPS.
        confidence_threshold (float): Confidence threshold for filtering detections. Defaults to 0.5.
        nms_threshold (float): Non-Maximum Suppression threshold. Defaults to 0.4.
        input_size (Tuple[int, int]): Input image size (width, height).
            Defaults to (640, 640).
            Note: Non-default sizes may cause slower inference and CoreML compatibility issues.
        **kwargs: Reserved for future advanced options.

    Attributes:
        model_name (SCRFDWeights): Selected model variant.
        confidence_threshold (float): Threshold used to filter low-confidence detections.
        nms_threshold (float): Threshold used during NMS to suppress overlapping boxes.
        input_size (Tuple[int, int]): Image size to which inputs are resized before inference.
        _num_feature_maps (int): Number of feature map levels used in the model.
        _feat_stride_fpn (List[int]): Feature map strides corresponding to each detection level.
        _num_anchors (int): Number of anchors per feature location.
        _center_cache (Dict): Cached anchor centers for efficient forward passes.
        _model_path (str): Absolute path to the downloaded/verified model weights.

    Raises:
        ValueError: If the model weights are invalid or not found.
        RuntimeError: If the ONNX model fails to load or initialize.
    """

    def __init__(
        self,
        *,
        model_name: SCRFDWeights = SCRFDWeights.SCRFD_10G_KPS,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: tuple[int, int] = (640, 640),
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            input_size=input_size,
            **kwargs,
        )
        self._supports_landmarks = True  # SCRFD supports landmarks

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        # ------- SCRFD model params ------
        self._num_feature_maps = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self._center_cache = {}
        # ---------------------------------

        Logger.info(
            f'Initializing SCRFD with model={self.model_name}, confidence_threshold={self.confidence_threshold}, '
            f'nms_threshold={self.nms_threshold}, input_size={self.input_size}'
        )

        # Get path to model weights
        self._model_path = verify_model_weights(self.model_name)
        Logger.info(f'Verified model weights located at: {self._model_path}')

        # Initialize model
        self._initialize_model(self._model_path)

    def _initialize_model(self, model_path: str) -> None:
        """Initialize an ONNX model session from the given path.

        Args:
            model_path: The file path to the ONNX model.

        Raises:
            RuntimeError: If the model fails to load.
        """
        try:
            self.session = create_onnx_session(model_path)
            self.input_names = self.session.get_inputs()[0].name
            self.output_names = [x.name for x in self.session.get_outputs()]
            Logger.info(f'Successfully initialized the model from {model_path}')
        except Exception as e:
            Logger.error(f"Failed to load model from '{model_path}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize model session for '{model_path}'") from e

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference.

        Args:
            image: Input image with shape (H, W, C).

        Returns:
            Preprocessed image tensor with shape (1, C, H, W).
        """
        image = image.astype(np.float32)
        image = (image - 127.5) / 127.5
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = np.expand_dims(image, axis=0)

        return image

    def inference(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        """Perform model inference on the preprocessed image tensor.

        Args:
            input_tensor: Preprocessed input tensor with shape (1, C, H, W).

        Returns:
            List of raw model outputs.
        """
        return self.session.run(self.output_names, {self.input_names: input_tensor})

    def postprocess(
        self,
        outputs: list[np.ndarray],
        image_size: tuple[int, int],
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """Process model outputs into detection results.

        Args:
            outputs: Raw outputs from the detection model.
            image_size: Size of the input image as (height, width).

        Returns:
            Tuple of (scores_list, bboxes_list, landmarks_list).
        """
        scores_list: list[np.ndarray] = []
        bboxes_list = []
        kpss_list = []

        image_size = image_size

        num_feature_maps = self._num_feature_maps
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outputs[idx]
            bbox_preds = outputs[num_feature_maps + idx] * stride
            kps_preds = outputs[2 * num_feature_maps + idx] * stride

            # Generate anchors
            fm_height = image_size[0] // stride
            fm_width = image_size[1] // stride
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

            pos_indices = np.where(scores >= self.confidence_threshold)[0]
            if len(pos_indices) == 0:
                continue

            bboxes = distance2bbox(anchor_centers, bbox_preds)[pos_indices]
            scores_selected = scores[pos_indices]
            scores_list.append(scores_selected)
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

        image, resize_factor = resize_image(image, target_shape=self.input_size)

        image_tensor = self.preprocess(image)

        # ONNXRuntime inference
        outputs = self.inference(image_tensor)

        scores_list, bboxes_list, kpss_list = self.postprocess(outputs, image_size=image.shape[:2])

        # Handle case when no faces are detected
        if not scores_list:
            return []

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        bboxes = np.vstack(bboxes_list) / resize_factor
        landmarks = np.vstack(kpss_list) / resize_factor

        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        keep = non_max_suppression(pre_det, threshold=self.nms_threshold)

        detections = pre_det[keep, :]
        landmarks = landmarks[order, :, :]
        landmarks = landmarks[keep, :, :].astype(np.float32)

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
