# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from uniface.face_utils import face_alignment
from uniface.log import Logger
from uniface.onnx_utils import create_onnx_session

__all__ = ['BaseRecognizer', 'PreprocessConfig']


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing images before feeding them into the model.

    Attributes:
        input_mean: Mean value(s) for normalization.
        input_std: Standard deviation value(s) for normalization.
        input_size: Target image size as (height, width).
    """

    input_mean: float | list[float] = 127.5
    input_std: float | list[float] = 127.5
    input_size: tuple[int, int] = (112, 112)


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
            self.session = create_onnx_session(self.model_path)

            # Extract input configuration
            input_cfg = self.session.get_inputs()[0]
            self.input_name = input_cfg.name

            # Verify input dimensions match our configuration
            input_shape = input_cfg.shape
            model_input_size = tuple(input_shape[2:4][::-1])  # (width, height)
            if model_input_size != self.input_size:
                Logger.warning(f'Model input size {model_input_size} differs from configured size {self.input_size}')

            # Extract output configuration
            self.output_names = [output.name for output in self.session.get_outputs()]
            self.output_shape = self.session.get_outputs()[0].shape

            assert len(self.output_names) == 1, 'Expected only one output node.'
            Logger.info(f'Successfully initialized face encoder from {self.model_path}')

        except Exception as e:
            Logger.error(
                f"Failed to load face encoder model from '{self.model_path}'",
                exc_info=True,
            )
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

        if isinstance(self.input_std, list | tuple):
            # Per-channel normalization
            rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB).astype(np.float32)
            normalized_img = (rgb_img - np.array(self.input_mean, dtype=np.float32)) / np.array(
                self.input_std, dtype=np.float32
            )

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
                swapRB=True,  # Convert BGR to RGB
            )

        return blob

    def get_embedding(self, image: np.ndarray, landmarks: np.ndarray | None = None) -> np.ndarray:
        """Extract face embedding from an image.

        Args:
            image: Input face image in BGR format. If already aligned (112x112),
                landmarks can be None.
            landmarks: Facial landmarks (5 points for alignment). Optional if
                image is already aligned.

        Returns:
            Face embedding vector (typically 512-dimensional).
        """
        # If landmarks are provided, align the face first
        if landmarks is not None:
            aligned_face, _ = face_alignment(image, landmarks, image_size=self.input_size)
        else:
            # Assume image is already aligned
            aligned_face = image

        # Generate embedding from aligned face
        face_blob = self.preprocess(aligned_face)
        embedding = self.session.run(self.output_names, {self.input_name: face_blob})[0]

        return embedding

    def get_normalized_embedding(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Extract an L2-normalized face embedding vector from an image.

        Args:
            image: Input face image in BGR format.
            landmarks: Facial landmarks (5 points for alignment).

        Returns:
            L2-normalized face embedding vector (typically 512-dimensional).
        """
        embedding = self.get_embedding(image, landmarks)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def __call__(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Callable shortcut for the `get_normalized_embedding` method.

        Args:
            image: Input face image in BGR format.
            landmarks: Facial landmarks (5 points for alignment).

        Returns:
            L2-normalized face embedding vector (typically 512-dimensional).
        """
        return self.get_normalized_embedding(image, landmarks)
