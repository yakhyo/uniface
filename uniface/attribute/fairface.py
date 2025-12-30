# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


import cv2
import numpy as np

from uniface.attribute.base import Attribute
from uniface.constants import FairFaceWeights
from uniface.log import Logger
from uniface.model_store import verify_model_weights
from uniface.onnx_utils import create_onnx_session
from uniface.types import AttributeResult

__all__ = ['AGE_LABELS', 'RACE_LABELS', 'FairFace']

# Label definitions
RACE_LABELS = [
    'White',
    'Black',
    'Latino Hispanic',
    'East Asian',
    'Southeast Asian',
    'Indian',
    'Middle Eastern',
]
AGE_LABELS = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']


class FairFace(Attribute):
    """
    FairFace attribute prediction model using ONNX Runtime.

    This class inherits from the base `Attribute` class and implements the
    functionality for predicting race (7 categories), gender (2 categories),
    and age (9 groups) from a face image. It requires a bounding box to locate the face.

    The model is trained on the FairFace dataset which provides balanced demographics
    for more equitable predictions across different racial and gender groups.

    Args:
        model_name (FairFaceWeights): The enum specifying the model weights to load.
            Defaults to `FairFaceWeights.DEFAULT`.
        input_size (Optional[Tuple[int, int]]): Input size (height, width).
            If None, defaults to (224, 224). Defaults to None.
    """

    def __init__(
        self,
        model_name: FairFaceWeights = FairFaceWeights.DEFAULT,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        """
        Initializes the FairFace prediction model.

        Args:
            model_name (FairFaceWeights): The enum specifying the model weights to load.
            input_size (Optional[Tuple[int, int]]): Input size (height, width).
                If None, defaults to (224, 224).
        """
        Logger.info(f'Initializing FairFace with model={model_name.name}')
        self.model_path = verify_model_weights(model_name)
        self.input_size = input_size if input_size is not None else (224, 224)
        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Initializes the ONNX model and creates an inference session.
        """
        try:
            self.session = create_onnx_session(self.model_path)
            # Get model input details from the loaded model
            input_meta = self.session.get_inputs()[0]
            self.input_name = input_meta.name
            self.output_names = [output.name for output in self.session.get_outputs()]
            Logger.info(f'Successfully initialized FairFace model with input size {self.input_size}')
        except Exception as e:
            Logger.error(
                f"Failed to load FairFace model from '{self.model_path}'",
                exc_info=True,
            )
            raise RuntimeError(f'Failed to initialize FairFace model: {e}') from e

    def preprocess(self, image: np.ndarray, bbox: list | np.ndarray | None = None) -> np.ndarray:
        """
        Preprocesses the face image for inference.

        Args:
            image (np.ndarray): The input image in BGR format.
            bbox (Optional[Union[List, np.ndarray]]): Face bounding box [x1, y1, x2, y2].
                If None, uses the entire image.

        Returns:
            np.ndarray: The preprocessed image blob ready for inference.
        """
        # Crop face if bbox provided
        if bbox is not None:
            bbox = np.asarray(bbox, dtype=int)
            x1, y1, x2, y2 = bbox[:4]

            # Add padding (25% of face size)
            w, h = x2 - x1, y2 - y1
            padding = 0.25
            x_pad = int(w * padding)
            y_pad = int(h * padding)

            x1 = max(0, x1 - x_pad)
            y1 = max(0, y1 - y_pad)
            x2 = min(image.shape[1], x2 + x_pad)
            y2 = min(image.shape[0], y2 + y_pad)

            image = image[y1:y2, x1:x2]

        # Resize to input size (width, height for cv2.resize)
        image = cv2.resize(image, self.input_size[::-1])

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize with ImageNet mean and std
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        # Transpose to CHW format and add batch dimension
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)

        return image

    def postprocess(self, prediction: tuple[np.ndarray, np.ndarray, np.ndarray]) -> AttributeResult:
        """
        Processes the raw model output to extract race, gender, and age.

        Args:
            prediction (Tuple[np.ndarray, np.ndarray, np.ndarray]): Raw outputs from model
                (race_logits, gender_logits, age_logits).

        Returns:
            AttributeResult: Result containing gender (0=Female, 1=Male), age_group, and race.
        """
        race_logits, gender_logits, age_logits = prediction

        # Apply softmax
        race_probs = self._softmax(race_logits[0])
        gender_probs = self._softmax(gender_logits[0])
        age_probs = self._softmax(age_logits[0])

        # Get predictions
        race_idx = int(np.argmax(race_probs))
        raw_gender_idx = int(np.argmax(gender_probs))
        age_idx = int(np.argmax(age_probs))

        # Normalize gender: model outputs 0=Male, 1=Female → standard 0=Female, 1=Male
        gender = 1 - raw_gender_idx

        return AttributeResult(
            gender=gender,
            age_group=AGE_LABELS[age_idx],
            race=RACE_LABELS[race_idx],
        )

    def predict(self, image: np.ndarray, bbox: list | np.ndarray | None = None) -> AttributeResult:
        """
        Predicts race, gender, and age for a face.

        Args:
            image (np.ndarray): The input image in BGR format.
            bbox (Optional[Union[List, np.ndarray]]): Face bounding box [x1, y1, x2, y2].
                If None, uses the entire image.

        Returns:
            AttributeResult: Result containing:
                - gender: 0=Female, 1=Male
                - age_group: Age range string like "20-29"
                - race: Race/ethnicity label
        """
        # Preprocess
        input_blob = self.preprocess(image, bbox)

        # Inference
        outputs = self.session.run(self.output_names, {self.input_name: input_blob})

        # Postprocess
        return self.postprocess(outputs)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax values for numerical stability."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
