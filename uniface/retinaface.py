# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import os
import cv2
import numpy as np
import onnxruntime as ort

import torch
from typing import Tuple, List, Optional, Literal

from uniface.log import Logger
from uniface.model_store import verify_model_weights

from uniface.common import (
    nms,
    resize_image,
    decode_boxes,
    generate_anchors,
    decode_landmarks
)


class RetinaFace:
    """
    A class for face detection using the RetinaFace model.

    Args:
        model (str): Path or identifier of the model weights.
        conf_thresh (float): Confidence threshold for detections. Defaults to 0.5.
        nms_thresh (float): Non-maximum suppression threshold. Defaults to 0.4.
        pre_nms_topk (int): Maximum number of detections before NMS. Defaults to 5000.
        post_nms_topk (int): Maximum number of detections after NMS. Defaults to 750.
        dynamic_size (Optional[bool]): Whether to adjust anchor generation dynamically based on image size. Defaults to False.
        input_size (Optional[Tuple[int, int]]): Static input size for the model (width, height). Defaults to (640, 640).

    Attributes:
        conf_thresh (float): Confidence threshold for filtering detections.
        nms_thresh (float): Threshold for NMS to remove duplicate detections.
        pre_nms_topk (int): Maximum detections to consider before applying NMS.
        post_nms_topk (int): Maximum detections retained after applying NMS.
        dynamic_size (bool): Indicates if input size and anchors are dynamically adjusted.
        input_size (Tuple[int, int]): The model's input image size.
        _model_path (str): Path to the model weights.
        _priors (torch.Tensor): Precomputed anchor boxes for static input size.
    """

    def __init__(
        self,
        model: str,
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        pre_nms_topk: int = 5000,
        post_nms_topk: int = 750,
        dynamic_size: Optional[bool] = False,
        input_size: Optional[Tuple[int, int]] = (640, 640),  # Default input size if dynamic_size=False
    ) -> None:

        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.dynamic_size = dynamic_size
        self.input_size = input_size

        Logger.info(
            f"Initializing RetinaFace with model={model}, conf_thresh={conf_thresh}, nms_thresh={nms_thresh}, "
            f"pre_nms_topk={pre_nms_topk}, post_nms_topk={post_nms_topk}, dynamic_size={dynamic_size}, "
            f"input_size={input_size}"
        )

        # Get path to model weights
        self._model_path = verify_model_weights(model)
        Logger.info(f"Verified model weights located at: {self._model_path}")

        # Precompute anchors if using static size
        if not dynamic_size and input_size is not None:
            self._priors = generate_anchors(image_size=input_size)
            Logger.debug("Generated anchors for static input size.")

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
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            Logger.info(f"Successfully initialized the model from {model_path}")
        except Exception as e:
            Logger.error(f"Failed to load model from '{model_path}': {e}")
            raise RuntimeError(f"Failed to initialize model session for '{model_path}'") from e

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess input image for model inference.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Preprocessed image tensor with shape (1, C, H, W)
        """
        image = np.float32(image) - np.array([104, 117, 123], dtype=np.float32)
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = np.expand_dims(image, axis=0)  # Add batch dimension (1, C, H, W)
        return image

    def inference(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """Perform model inference on the preprocessed image tensor.

        Args:
            input_tensor (np.ndarray): Preprocessed input tensor.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Raw model outputs.
        """
        return self.session.run(None, {self.input_name: input_tensor})

    def detect(
        self,
        image: np.ndarray,
        max_num: Optional[int] = 0,
        metric: Literal["default", "max"] = "default",
        center_weight: Optional[float] = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform face detection on an input image and return bounding boxes and landmarks.

        Args:
            image (np.ndarray): Input image as a NumPy array of shape (height, width, channels).
            max_num (int, optional): Maximum number of detections to return. Defaults to 1.
            metric (str, optional): Metric for ranking detections when `max_num` is specified. 
                Options:
                - "default": Prioritize detections closer to the image center.
                - "max": Prioritize detections with larger bounding box areas.
            center_weight (float, optional): Weight for penalizing detections farther from the image center 
                when using the "default" metric. Defaults to 2.0.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Detection results containing:
                - detections (np.ndarray): Array of detected bounding boxes with confidence scores.
                Shape: (num_detections, 5), where each row is [x_min, y_min, x_max, y_max, score].
                - landmarks (np.ndarray): Array of detected facial landmarks.
                Shape: (num_detections, 5, 2), where each row contains 5 landmark points (x, y).
        """

        if self.dynamic_size:
            height, width, _ = image.shape
            self._priors = generate_anchors(image_size=(height, width))  # generate anchors for each input image
            resize_factor = 1.0  # No resizing
        else:
            image, resize_factor = resize_image(image, target_shape=self.input_size)

        height, width, _ = image.shape
        image_tensor = self.preprocess(image)

        # ONNXRuntime inference
        outputs = self.inference(image_tensor)

        # Postprocessing
        detections, landmarks = self.postprocess(outputs, resize_factor, shape=(width, height))

        if max_num > 0 and detections.shape[0] > max_num:
            # Calculate area of detections
            areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])

            # Calculate offsets from image center
            center = (height // 2, width // 2)
            offsets = np.vstack([
                (detections[:, 0] + detections[:, 2]) / 2 - center[1],
                (detections[:, 1] + detections[:, 3]) / 2 - center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), axis=0)

            # Calculate scores based on the chosen metric
            if metric == 'max':
                scores = areas
            else:
                scores = areas - offset_dist_squared * center_weight

            # Sort by scores and select top `max_num`
            sorted_indices = np.argsort(scores)[::-1][:max_num]

            detections = detections[sorted_indices]
            landmarks = landmarks[sorted_indices]

        return detections, landmarks

    def postprocess(self, outputs: List[np.ndarray], resize_factor: float, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process the model outputs into final detection results.

        Args:
            outputs (List[np.ndarray]): Raw outputs from the detection model.
                - outputs[0]: Location predictions (bounding box coordinates).
                - outputs[1]: Class confidence scores.
                - outputs[2]: Landmark predictions.
            resize_factor (float): Factor used to resize the input image during preprocessing.
            shape (Tuple[int, int]): Original shape of the image as (height, width).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed results containing:
                - detections (np.ndarray): Array of detected bounding boxes with confidence scores.
                Shape: (num_detections, 5), where each row is [x_min, y_min, x_max, y_max, score].
                - landmarks (np.ndarray): Array of detected facial landmarks.
                Shape: (num_detections, 5, 2), where each row contains 5 landmark points (x, y).
        """
        loc, conf, landmarks = outputs[0].squeeze(0), outputs[1].squeeze(0), outputs[2].squeeze(0)

        # Decode boxes and landmarks
        boxes = decode_boxes(torch.tensor(loc), self._priors).cpu().numpy()
        landmarks = decode_landmarks(torch.tensor(landmarks), self._priors).cpu().numpy()

        boxes, landmarks = self._scale_detections(boxes, landmarks, resize_factor, shape=(shape[0], shape[1]))

        # Extract confidence scores for the face class
        scores = conf[:, 1]
        mask = scores > self.conf_thresh

        # Filter by confidence threshold
        boxes, landmarks, scores = boxes[mask], landmarks[mask], scores[mask]

        # Sort by scores
        order = scores.argsort()[::-1][:self.pre_nms_topk]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

        # Apply NMS
        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(detections, self.nms_thresh)
        detections, landmarks = detections[keep], landmarks[keep]

        # Keep top-k detections
        detections, landmarks = detections[:self.post_nms_topk], landmarks[:self.post_nms_topk]

        landmarks = landmarks.reshape(-1, 5, 2).astype(np.int32)

        return detections, landmarks

    def _scale_detections(self, boxes: np.ndarray, landmarks: np.ndarray, resize_factor: float, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Scale bounding boxes and landmarks to the original image size."""
        bbox_scale = np.array([shape[0], shape[1]] * 2)
        boxes = boxes * bbox_scale / resize_factor

        landmark_scale = np.array([shape[0], shape[1]] * 5)
        landmarks = landmarks * landmark_scale / resize_factor

        return boxes, landmarks
