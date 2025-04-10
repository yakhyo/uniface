# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo
# Modified from insightface repository

import os
import cv2
import numpy as np
import onnxruntime as ort

from typing import Tuple, Optional, Union, List
from dataclasses import dataclass

from uniface.log import Logger
from uniface.model_store import verify_model_weights
from uniface.face_utils import compute_similarity, face_alignment
from uniface.constants import SphereFaceWeights, MobileFaceWeights, ArcFaceWeights


__all__ = ["BaseFaceEncoder", "PreprocessConfig"]


@dataclass
class PreprocessConfig:
    """
    Configuration for preprocessing images before feeding them into the model.
    """
    input_mean: Union[float, List[float]] = 127.5
    input_std: Union[float, List[float]] = 127.5
    input_size: Tuple[int, int] = (112, 112)


class BaseFaceEncoder:
    """
    Unified Face Encoder supporting multiple model families (e.g., SphereFace, MobileFace).
    """

    def __init__(
        self,
        model_name: SphereFaceWeights | MobileFaceWeights | ArcFaceWeights = MobileFaceWeights.MNET_V2,
        preprocessing: PreprocessConfig = PreprocessConfig(),
    ) -> None:
        """
        Initializes the FaceEncoder model for inference.

        Args:
            model_name (SphereFaceWeights | MobileFaceWeights | ArcFaceWeights): Selected model weight enum.
            preprocessing (PreprocessConfig): Configuration for input normalization and resizing.
        """
        self.input_mean = preprocessing.input_mean
        self.input_std = preprocessing.input_std
        self.input_size = preprocessing.input_size

        Logger.info(
            f"Initializing Face Recognition with model={model_name}, "
            f"input_mean={self.input_mean}, input_std={self.input_std}, input_size={self.input_size}"
        )

        # Get path to model weights
        self._model_path = verify_model_weights(model_name)
        Logger.info(f"Verfied model weights located at: {self._model_path}")

        # Initialize model
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
        model_input_size = tuple(input_shape[2:4][::-1])  # (width, height)

        if model_input_size != self.input_size:
            Logger.warning(f"Model input size {model_input_size} differs from configured size {self.input_size}")

        self.input_name = input_cfg.name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.output_shape = self.session.get_outputs()[0].shape

        assert len(self.output_names) == 1, "Expected only one output node."

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image: resize, normalize, and convert it to a blob.

        Args:
            image (np.ndarray): Input image in BGR format.

        Returns:
            np.ndarray: Preprocessed image as a NumPy array ready for inference.
        """
        image = cv2.resize(image, self.input_size)  # Resize to (112, 112)
        if isinstance(self.input_std, (list, tuple)):
            # if self.input_std is a list, we assume it's per-channel std
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

            image -= np.array(self.input_mean, dtype=np.float32)
            image /= np.array(self.input_std, dtype=np.float32)

            # Change to NCHW (batch, channels, height, width)
            blob = np.transpose(image, (2, 0, 1))  # CHW
            blob = np.expand_dims(blob, axis=0)  # NCHW
        else:
            # cv2.dnn.blobFromImage does not support per-channel std so we use a single value here
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
        aligned_face, _ = face_alignment(image, landmarks)  # Use your function for alignment
        blob = self.preprocess(aligned_face)  # Convert to blob
        embedding = self.session.run(self.output_names, {self.input_name: blob})[0]
        return embedding  # Return the 512-D feature vector
