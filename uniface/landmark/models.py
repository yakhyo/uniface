# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple

from uniface.log import Logger
from uniface.constants import LandmarkWeights
from uniface.model_store import verify_model_weights
from uniface.face_utils import bbox_center_alignment, transform_points_2d
from .base import BaseLandmarker

__all__ = ['Landmark']


class Landmark106(BaseLandmarker):
    """Facial landmark model for predicting 106 facial keypoints.

    This class implements the BaseLandmarker and provides an end-to-end
    pipeline for 106-point facial landmark detection. It handles model
    loading, preprocessing of a face crop based on a bounding box,
    inference, and post-processing to map landmarks back to the
    original image coordinates.

    Args:
        model_name (LandmarkWeights): The enum specifying the landmark model to load.
            Defaults to `LandmarkWeights.DEFAULT`.
        input_size (Tuple[int, int]): The resolution (width, height) for the model's
            input. Defaults to (192, 192).

    Example:
        >>> # Assume 'image' is a loaded image and 'bbox' is a face bounding box
        >>> # bbox = [x1, y1, x2, y2]
        >>>
        >>> landmarker = Landmark106()
        >>> landmarks = landmarker.get_landmarks(image, bbox)
        >>> print(landmarks.shape)
        (106, 2)
    """
    def __init__(
        self,
        model_name: LandmarkWeights = LandmarkWeights.DEFAULT,
        input_size: Tuple[int, int] = (192, 192)
    ) -> None:
        Logger.info(
            f"Initializing Facial Landmark with model={model_name}, "
            f"input_size={input_size}"
        )
        self.input_size = input_size
        self.input_std = 1.0
        self.input_mean = 0.0
        self.model_path = verify_model_weights(model_name)
        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize the ONNX model from the stored model path.

        Raises:
            RuntimeError: If the model fails to load or initialize.
        """
        try:
            self.session = ort.InferenceSession(
                self.model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )

            # Get input configuration
            input_metadata = self.session.get_inputs()[0]
            input_shape = input_metadata.shape
            self.input_size = tuple(input_shape[2:4][::-1])  # Update input size from model

            # Get input/output names
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]

            # Determine landmark dimensions from output shape
            output_shape = self.session.get_outputs()[0].shape
            self.lmk_dim = 2  # x,y coordinates
            self.lmk_num = output_shape[1] // self.lmk_dim  # Number of landmarks

            Logger.info(f"Model initialized with {self.lmk_num} landmarks")

        except Exception as e:
            Logger.error(f"Failed to load landmark model from '{self.model_path}'", exc_info=True)
            raise RuntimeError(f"Failed to initialize landmark model: {e}")

    def preprocess(self, image: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepares a face crop for inference.

        This method takes a face bounding box, performs a center alignment to
        warp the face into the model's required input size, and then creates
        a normalized blob ready for the ONNX session.

        Args:
            image (np.ndarray): The full source image in BGR format.
            bbox (np.ndarray): The bounding box of the face [x1, y1, x2, y2].

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - The preprocessed image blob ready for inference.
                - The affine transformation matrix used for alignment.
        """
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        scale = self.input_size[0] / (max(width, height) * 1.5)
        
        aligned_face, transform_matrix = bbox_center_alignment(image, center, self.input_size[0], scale, 0.0)

        face_blob = cv2.dnn.blobFromImage(
            aligned_face, 1.0 / self.input_std, self.input_size,
            (self.input_mean, self.input_mean, self.input_mean), swapRB=True
        )
        return face_blob, transform_matrix

    def postprocess(self, predictions: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """Converts raw model predictions back to original image coordinates.

        This method reshapes the model's flat output array into landmark points,
        denormalizes them to the model's input space, and then applies an
        inverse affine transformation to map them back to the original image space.

        Args:
            predictions (np.ndarray): Raw landmark coordinates from the model output.
            transform_matrix (np.ndarray): The affine transformation matrix from preprocessing.

        Returns:
            np.ndarray: An array of landmark points in the original image's coordinates.
        """
        landmarks = predictions.reshape((-1, 2))
        landmarks[:, 0:2] += 1
        landmarks[:, 0:2] *= (self.input_size[0] // 2)
        
        inverse_matrix = cv2.invertAffineTransform(transform_matrix)
        landmarks = transform_points_2d(landmarks, inverse_matrix)
        return landmarks

    def get_landmarks(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Predicts facial landmarks for the given image and face bounding box.

        This is the main public method that orchestrates the full pipeline of
        preprocessing, inference, and post-processing.

        Args:
            image (np.ndarray): The full source image in BGR format.
            bbox (np.ndarray): A bounding box of a face [x1, y1, x2, y2].

        Returns:
            np.ndarray: An array of predicted landmark points with shape (106, 2).
        """
        face_blob, transform_matrix = self.preprocess(image, bbox)
        raw_predictions = self.session.run(
            self.output_names, {self.input_names[0]: face_blob}
        )[0][0]
        landmarks = self.postprocess(raw_predictions, transform_matrix)
        return landmarks



# TODO: For testing purposes only, remote later
if __name__ == "__main__":
    # UPDATED: Use the high-level factory functions
    from uniface.detection import create_detector
    from uniface.landmark import create_landmarker

    # 1. Create the detector and landmarker using the new API
    face_detector = create_detector('retinaface')
    landmarker = create_landmarker() # Uses the default '2d106det' method

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not available.")
        exit()

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed.")
            break

        # 2. The detect method returns a list of dictionaries
        faces = face_detector.detect(frame)

        if not faces:
            cv2.imshow("Facial Landmark Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # 3. Loop through the list of face dictionaries
        for face in faces:
            # Extract the bounding box
            bbox = face['bbox']
            
            # 4. Get landmarks for the current face using its bounding box
            landmarks = landmarker.get_landmarks(frame, bbox)

            # --- Drawing Logic ---
            # Draw the landmarks
            for (x, y) in landmarks.astype(int):
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Draw the bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow("Facial Landmark Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
