# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from abc import ABC, abstractmethod
import cv2
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from typing import Tuple, Union, List

from uniface.log import Logger
from uniface.face_utils import face_alignment


@dataclass
class PreprocessConfig:
    """
    Configuration for preprocessing images before feeding them into the model.
    """
    input_mean: Union[float, List[float]] = 127.5
    input_std: Union[float, List[float]] = 127.5
    input_size: Tuple[int, int] = (112, 112)


class BaseRecognizer(ABC):
    """
    Abstract Base Class for all face recognition models.
    It provides the core functionality for preprocessing, inference, and embedding extraction.
    """
    @abstractmethod
    def __init__(self, model_path: str, preprocessing: PreprocessConfig) -> None:
        """
        Initializes the model. Subclasses must call this.

        Args:
            model_path (str): The direct path to the verified ONNX model.
            preprocessing (PreprocessConfig): The configuration for preprocessing.
        """
        self.input_mean = preprocessing.input_mean
        self.input_std = preprocessing.input_std
        self.input_size = preprocessing.input_size

        self.model_path = model_path
        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Loads the ONNX model and prepares it for inference.

        Raises:
            RuntimeError: If the model fails to load or initialize.
        """
        try:
            # Initialize model session with available providers
            self.session = ort.InferenceSession(
                self.model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )

            # Extract input configuration
            input_cfg = self.session.get_inputs()[0]
            self.input_name = input_cfg.name

            # Verify input dimensions match our configuration
            input_shape = input_cfg.shape
            model_input_size = tuple(input_shape[2:4][::-1])  # (width, height)
            if model_input_size != self.input_size:
                Logger.warning(f"Model input size {model_input_size} differs from configured size {self.input_size}")

            # Extract output configuration
            self.output_names = [output.name for output in self.session.get_outputs()]
            self.output_shape = self.session.get_outputs()[0].shape

            assert len(self.output_names) == 1, "Expected only one output node."
            Logger.info(f"Successfully initialized face encoder from {self.model_path}")

        except Exception as e:
            Logger.error(f"Failed to load face encoder model from '{self.model_path}'", exc_info=True)
            raise RuntimeError(f"Failed to initialize model session for '{self.model_path}'") from e

    def preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """
        Preprocess the image: resize, normalize, and convert it to a blob.

        Args:
            face_img: Input image in BGR format.

        Returns:
            Preprocessed image as a NumPy array ready for inference.
        """
        resized_img = cv2.resize(face_img, self.input_size)

        if isinstance(self.input_std, (list, tuple)):
            # Per-channel normalization
            rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB).astype(np.float32)
            normalized_img = (rgb_img - np.array(self.input_mean, dtype=np.float32)) / \
                np.array(self.input_std, dtype=np.float32)

            # Change to NCHW (batch, channels, height, width)
            blob = np.transpose(normalized_img, (2, 0, 1))  # CHW
            blob = np.expand_dims(blob, axis=0)  # NCHW
        else:
            # Single-value normalization
            blob = cv2.dnn.blobFromImage(
                resized_img,
                scalefactor=1.0 / self.input_std,
                size=self.input_size,
                mean=(self.input_mean, self.input_mean, self.input_mean),
                swapRB=True  # Convert BGR to RGB
            )

        return blob

    def get_embedding(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Extracts face embedding from an image.

        Args:
            image: Input face image (BGR format).
            landmarks: Facial landmarks (5 points for alignment).

        Returns:
            Face embedding vector (typically 512-dimensional).
        """
        # Align face using landmarks
        aligned_face, _ = face_alignment(image, landmarks)

        # Generate embedding from aligned face
        face_blob = self.preprocess(aligned_face)
        embedding = self.session.run(self.output_names, {self.input_name: face_blob})[0]

        return embedding

    def get_normalized_embedding(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Extracts a l2 normalized face embedding vector from an image.

        Args:
            image: Input face image (BGR format).
            landmarks: Facial landmarks (5 points for alignment).

        Returns:
            Normalized face embedding vector (typically 512-dimensional).
        """
        embedding = self.get_embedding(image, landmarks)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding
