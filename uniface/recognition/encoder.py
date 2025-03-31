# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import os
import cv2
import numpy as np
import onnxruntime as ort

from typing import Tuple, List, Optional, Literal

from uniface.face_utils import compute_similarity, face_alignment
from uniface.model_store import verify_model_weights
from uniface.constants import FaceEncoderWeights
from uniface.logger import Logger


class FaceEncoder:
    """
    Face recognition model using ONNX Runtime for inference and OpenCV for image preprocessing,
    utilizing an external face alignment function.
    """

    def __init__(
        self,
        model_path: Optional[FaceEncoderWeights] = FaceEncoderWeights.MNET_V2,
    ) -> None:
        """
        Initializes the FaceEncoder model for inference.

        Args:
            model_path (str): Path to the ONNX model file.
        """
        self.input_mean = 127.5
        self.input_std = 127.5

        # Get path to model weights
        self._model_path = verify_model_weights(model_path)
        Logger.info(f"Verfied model weights located at: {self._model_path}")

        self._initialize_model(self._model_path)

    def _initialize_model(self, model_path: str) -> None:
        """
        Loads the ONNX model and prepares it for inference.

        Args:
            model_path (str): Path to the ONNX model file.

        Raises:
            RuntimeError: If the model fails to load or initialize.
        """
        try:
            self.session = ort.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self._setup_model()
            Logger.info(f"Successfully initialized face encoder from {model_path}")
        except Exception as e:
            Logger.error(f"Failed to load face encoder model from '{model_path}'", exc_info=True)
            raise RuntimeError(f"Failed to initialize model session for '{model_path}'") from e

    def _setup_model(self) -> None:
        """
        Extracts input/output configuration from the ONNX model session.
        """
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape

        self.input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])  # (width, height)

        outputs = self.session.get_outputs()
        self.output_names = [output.name for output in outputs]

        assert len(self.output_names) == 1, "Expected only one output node."
        self.output_shape = outputs[0].shape

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image: resize, normalize, and convert it to a blob.

        Args:
            image (np.ndarray): Input image in BGR format.

        Returns:
            np.ndarray: Preprocessed image as a NumPy array ready for inference.
        """
        image = cv2.resize(image, self.input_size)  # Resize to (112, 112)
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / self.input_std,
            size=self.input_size,
            mean=(self.input_mean, self.input_mean, self.input_mean),
            swapRB=True  # Convert BGR to RGB
        )
        return blob

    def get_embedding(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Extracts face embedding from an aligned image.

        Args:
            image (np.ndarray): Input face image (BGR format).
            landmarks (np.ndarray): Facial landmarks (5 points for alignment).

        Returns:
            np.ndarray: 512-dimensional face embedding.
        """
        aligned_face = face_alignment(image, landmarks)  # Use your function for alignment
        blob = self.preprocess(image)  # Convert to blob
        embedding = self.session.run(self.output_names, {self.input_name: blob})[0]
        return embedding  # Return the 512-D feature vector
