# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from typing import Any, Dict, List, Literal, Tuple

import numpy as np

from uniface.common import (
    decode_boxes,
    decode_landmarks,
    generate_anchors,
    non_max_suppression,
    resize_image,
)
from uniface.constants import RetinaFaceWeights
from uniface.log import Logger
from uniface.model_store import verify_model_weights
from uniface.onnx_utils import create_onnx_session

from .base import BaseDetector


class RetinaFace(BaseDetector):
    """
    Face detector based on the RetinaFace architecture.

    Title: "RetinaFace: Single-stage Dense Face Localisation in the Wild"
    Paper: https://arxiv.org/abs/1905.00641
    Code: https://github.com/yakhyo/retinaface-pytorch

    Args:
        model_name (RetinaFaceWeights): Model weights to use. Defaults to `RetinaFaceWeights.MNET_V2`.
        conf_thresh (float): Confidence threshold for filtering detections. Defaults to 0.5.
        nms_thresh (float): Non-maximum suppression (NMS) IoU threshold. Defaults to 0.4.
        input_size (Tuple[int, int]): Fixed input size (width, height) if `dynamic_size=False`.
            Defaults to (640, 640).
            Note: Non-default sizes may cause slower inference and CoreML compatibility issues.
        **kwargs: Advanced options:
            pre_nms_topk (int): Number of top-scoring boxes considered before NMS. Defaults to 5000.
            post_nms_topk (int): Max number of detections kept after NMS. Defaults to 750.
            dynamic_size (bool): If True, generate anchors dynamically per input image. Defaults to False.

    Attributes:
        model_name (RetinaFaceWeights): Selected model variant.
        conf_thresh (float): Threshold for confidence-based filtering.
        nms_thresh (float): IoU threshold used for NMS.
        pre_nms_topk (int): Limit on proposals before applying NMS.
        post_nms_topk (int): Limit on retained detections after NMS.
        dynamic_size (bool): Flag indicating dynamic or static input sizing.
        input_size (Tuple[int, int]): Static input size if `dynamic_size=False`.
        _model_path (str): Absolute path to the verified model weights.
        _priors (np.ndarray): Precomputed anchor boxes (if static size).
        _supports_landmarks (bool): Indicates landmark prediction support.

    Raises:
        ValueError: If the model weights are invalid or not found.
        RuntimeError: If the ONNX model fails to load or initialize.
    """

    def __init__(
        self,
        *,
        model_name: RetinaFaceWeights = RetinaFaceWeights.MNET_V2,
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
        self._supports_landmarks = True  # RetinaFace supports landmarks

        self.model_name = model_name
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size

        # Advanced options from kwargs
        self.pre_nms_topk = kwargs.get('pre_nms_topk', 5000)
        self.post_nms_topk = kwargs.get('post_nms_topk', 750)
        self.dynamic_size = kwargs.get('dynamic_size', False)

        Logger.info(
            f'Initializing RetinaFace with model={self.model_name}, conf_thresh={self.conf_thresh}, '
            f'nms_thresh={self.nms_thresh}, input_size={self.input_size}'
        )

        # Get path to model weights
        self._model_path = verify_model_weights(self.model_name)
        Logger.info(f'Verified model weights located at: {self._model_path}')

        # Precompute anchors if using static size
        if not self.dynamic_size and self.input_size is not None:
            self._priors = generate_anchors(image_size=self.input_size)
            Logger.debug('Generated anchors for static input size.')

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
        return self.session.run(self.output_names, {self.input_names: input_tensor})

    def detect(
        self,
        image: np.ndarray,
        *,
        max_num: int = 0,
        metric: Literal['default', 'max'] = 'max',
        center_weight: float = 2.0,
    ) -> List[Dict[str, Any]]:
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
            List[Dict[str, Any]]: List of face detection dictionaries, each containing:
                - 'bbox' (np.ndarray): Bounding box coordinates with shape (4,) as [x1, y1, x2, y2]
                - 'confidence' (float): Detection confidence score (0.0 to 1.0)
                - 'landmarks' (np.ndarray): 5-point facial landmarks with shape (5, 2)

        Example:
            >>> faces = detector.detect(image)
            >>> for face in faces:
            ...     bbox = face['bbox']  # np.ndarray with shape (4,)
            ...     confidence = face['confidence']  # float
            ...     landmarks = face['landmarks']  # np.ndarray with shape (5, 2)
            ...     # Can pass landmarks directly to recognition
            ...     embedding = recognizer.get_normalized_embedding(image, landmarks)
        """

        original_height, original_width = image.shape[:2]

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
            center = (original_height // 2, original_width // 2)
            offsets = np.vstack(
                [
                    (detections[:, 0] + detections[:, 2]) / 2 - center[1],
                    (detections[:, 1] + detections[:, 3]) / 2 - center[0],
                ]
            )
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

        faces = []
        for i in range(detections.shape[0]):
            face_dict = {
                'bbox': detections[i, :4],
                'confidence': float(detections[i, 4]),
                'landmarks': landmarks[i],
            }
            faces.append(face_dict)

        return faces

    def postprocess(
        self, outputs: List[np.ndarray], resize_factor: float, shape: Tuple[int, int]
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
        loc, conf, landmarks = (
            outputs[0].squeeze(0),
            outputs[1].squeeze(0),
            outputs[2].squeeze(0),
        )

        # Decode boxes and landmarks
        boxes = decode_boxes(loc, self._priors)
        landmarks = decode_landmarks(landmarks, self._priors)

        boxes, landmarks = self._scale_detections(boxes, landmarks, resize_factor, shape=(shape[0], shape[1]))

        # Extract confidence scores for the face class
        scores = conf[:, 1]
        mask = scores > self.conf_thresh

        # Filter by confidence threshold
        boxes, landmarks, scores = boxes[mask], landmarks[mask], scores[mask]

        # Sort by scores
        order = scores.argsort()[::-1][: self.pre_nms_topk]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

        # Apply NMS
        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = non_max_suppression(detections, self.nms_thresh)
        detections, landmarks = detections[keep], landmarks[keep]

        # Keep top-k detections
        detections, landmarks = (
            detections[: self.post_nms_topk],
            landmarks[: self.post_nms_topk],
        )

        landmarks = landmarks.reshape(-1, 5, 2).astype(np.float32)

        return detections, landmarks

    def _scale_detections(
        self,
        boxes: np.ndarray,
        landmarks: np.ndarray,
        resize_factor: float,
        shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Scale bounding boxes and landmarks to the original image size.
        bbox_scale = np.array([shape[0], shape[1]] * 2)
        boxes = boxes * bbox_scale / resize_factor

        landmark_scale = np.array([shape[0], shape[1]] * 5)
        landmarks = landmarks * landmark_scale / resize_factor

        return boxes, landmarks


# TODO: below is only for testing, remove it later
def draw_bbox(frame, bbox, score, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = map(int, bbox)  # Unpack 4 bbox values
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, f'{score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_keypoints(frame, points, color=(0, 0, 255), radius=2):
    for x, y in points.astype(np.int32):
        cv2.circle(frame, (int(x), int(y)), radius, color, -1)


if __name__ == '__main__':
    import cv2

    detector = RetinaFace(model_name=RetinaFaceWeights.MNET_050)
    print(detector.get_info())
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print('Failed to open webcam.')
        exit()

    print("Webcam started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Failed to read frame.')
            break

        # Get face detections as list of dictionaries
        faces = detector.detect(frame)

        # Process each detected face
        for face in faces:
            # Extract bbox and landmarks from dictionary
            bbox = face['bbox']  # [x1, y1, x2, y2]
            landmarks = face['landmarks']  # [[x1, y1], [x2, y2], ...]
            confidence = face['confidence']

            # Pass bbox and confidence separately
            draw_bbox(frame, bbox, confidence)

            # Convert landmarks to numpy array format if needed
            if landmarks is not None and len(landmarks) > 0:
                # Convert list of [x, y] pairs to numpy array
                points = np.array(landmarks, dtype=np.float32)  # Shape: (5, 2)
                draw_keypoints(frame, points)

        # Display face count
        cv2.putText(
            frame,
            f'Faces: {len(faces)}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow('FaceDetection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
