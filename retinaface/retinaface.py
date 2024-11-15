import os
import cv2
import numpy as np
import onnxruntime as ort

import torch
from typing import Tuple, List

from .log import logger
from .model_store import verify_model_weights

from .common import (
    nms,
    resize_image,
    decode_boxes,
    generate_anchors,
    decode_landmarks
)


class RetinaFace:
    def __init__(
        self,
        model: str,
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        input_size: Tuple[int, int] = (640, 640),
        pre_nms_topk: int = 5000,
        post_nms_topk: int = 750,
    ) -> None:

        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size

        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk

        # Get path to model weights
        self._model_path = verify_model_weights(model)

        # Generate anchor boxes
        self._priors = generate_anchors(image_size=input_size)

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
            logger.info(f"Successfully initialized the model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from '{model_path}': {e}")
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

    def detect(self, image: np.ndarray, input_size: Tuple[int, int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform face detection on an input image and return bounding boxes and landmarks.

        Args:
            image (np.ndarray): Input image as a NumPy array of shape (height, width, channels).
            input_size (Tuple[int, int], optional): Target size for resizing the input image (width, height).
                If provided and different from `self.input_size`, new anchors will be generated.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Detection results containing:
                - detections (np.ndarray): Array of detected bounding boxes with confidence scores.
                Shape: (num_detections, 5), where each row is [x_min, y_min, x_max, y_max, score].
                - landmarks (np.ndarray): Array of detected facial landmarks.
                Shape: (num_detections, 5, 2), where each row contains 5 landmark points (x, y).
        """
        if input_size is not None and input_size != self.input_size:
            self._priors = generate_anchors(image_size=input_size)

        # Preprocessing
        image, resize_factor = resize_image(image, target_shape=input_size)
        height, width, _ = image.shape
        image_tensor = self.preprocess(image)

        # ONNXRuntime inference
        outputs = self.inference(image_tensor)

        # Postprocessing
        detections, landmarks = self.postprocess(outputs, resize_factor, shape=(width, height))

        return detections, landmarks

    def postprocess(
        self,
        outputs: List[np.ndarray],
        resize_factor: float,
        shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
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

    def _scale_detections(
        self,
        boxes: np.ndarray,
        landmarks: np.ndarray,
        resize_factor: float,
        shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scale bounding boxes and landmarks to the original image size."""
        bbox_scale = np.array([shape[0], shape[1]] * 2)
        boxes = boxes * bbox_scale / resize_factor

        landmark_scale = np.array([shape[0], shape[1]] * 5)
        landmarks = landmarks * landmark_scale / resize_factor

        return boxes, landmarks
