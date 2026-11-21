# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


import cv2
import numpy as np
import torch

from uniface.attribute.base import BaseAttribute
from uniface.common import validate_image
from uniface.constants import EmotionWeights
from uniface.face_utils import face_alignment
from uniface.log import Logger
from uniface.model_store import verify_model_weights
from uniface.types import EmotionResult, Face

__all__ = ['Emotion']


class Emotion(BaseAttribute):
    """Emotion recognition model using a TorchScript model.

    This class inherits from the `BaseAttribute` base class and implements the
    functionality for predicting one of several emotion categories from a face
    image. It requires 5-point facial landmarks for alignment.

    Raises:
        ValueError: If the model weights are invalid or not found.
        RuntimeError: If the TorchScript model fails to load or initialize.
    """

    def __init__(
        self,
        *,
        model_name: EmotionWeights = EmotionWeights.AFFECNET7,
        input_size: tuple[int, int] = (112, 112),
        device: str | torch.device | None = None,
    ) -> None:
        """Initializes the emotion recognition model.

        Args:
            model_name (EmotionWeights): The enum for the model weights to load.
            input_size (tuple[int, int]): Expected input size as (height, width).
            device (str | torch.device | None): Torch device override (e.g. 'cpu', 'cuda').
                If None, auto-selects MPS > CUDA > CPU.
        """
        Logger.info(f'Initializing Emotion with model={model_name}')

        if device is not None:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.input_size = tuple(input_size)
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.model_path = verify_model_weights(model_name)

        # Define emotion labels based on the selected model
        self.emotion_labels = [
            'Neutral',
            'Happy',
            'Sad',
            'Surprise',
            'Fear',
            'Disgust',
            'Angry',
        ]
        if model_name == EmotionWeights.AFFECNET8:
            self.emotion_labels.append('Contempt')

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Loads and initializes the TorchScript model for inference."""
        try:
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
            # Warm-up with a dummy input for faster first inference
            dummy_input = torch.randn(1, 3, *self.input_size).to(self.device)
            with torch.no_grad():
                self.model(dummy_input)
            Logger.info(f'Successfully initialized Emotion model on {self.device}')
        except Exception as e:
            Logger.error(f"Failed to load Emotion model from '{self.model_path}'", exc_info=True)
            raise RuntimeError(f'Failed to initialize Emotion model: {e}') from e

    def preprocess(self, image: np.ndarray, landmarks: list | np.ndarray) -> torch.Tensor:
        """Aligns the face using landmarks and preprocesses it into a tensor.

        Args:
            image (np.ndarray): The full input image in BGR format.
            landmarks (list | np.ndarray): The 5-point facial landmarks.

        Returns:
            torch.Tensor: The preprocessed image tensor ready for inference.
        """
        validate_image(image)
        landmarks = np.asarray(landmarks)

        aligned_image, _ = face_alignment(image, landmarks)

        # Convert BGR to RGB, resize only if input_size != (112, 112), normalize, and convert to NCHW
        rgb_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)
        if rgb_image.shape[:2] != self.input_size:
            rgb_image = cv2.resize(rgb_image, self.input_size[::-1])
        resized_image = rgb_image.astype(np.float32) / 255.0
        normalized_image = (resized_image - self._mean) / self._std
        transposed_image = np.ascontiguousarray(normalized_image.transpose((2, 0, 1)), dtype=np.float32)

        return torch.from_numpy(transposed_image).unsqueeze(0).to(self.device)

    def postprocess(self, prediction: torch.Tensor) -> EmotionResult:
        """Processes the raw model output to get the emotion label and confidence score."""
        probabilities = torch.nn.functional.softmax(prediction.detach(), dim=-1).reshape(-1).cpu().numpy()
        pred_index = int(np.argmax(probabilities))
        emotion_label = self.emotion_labels[pred_index]
        confidence = float(probabilities[pred_index])
        return EmotionResult(emotion=emotion_label, confidence=confidence)

    def predict(self, image: np.ndarray, face: Face) -> EmotionResult:
        """Predict emotion and enrich the Face in-place.

        Args:
            image: The full input image in BGR format.
            face: Detected face; `face.landmarks` is used for alignment.

        Returns:
            `EmotionResult` with emotion label and confidence score.
        """
        input_tensor = self.preprocess(image, face.landmarks)
        with torch.no_grad():
            output = self.model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]

        result = self.postprocess(output)
        face.emotion = result.emotion
        face.emotion_confidence = result.confidence
        return result
