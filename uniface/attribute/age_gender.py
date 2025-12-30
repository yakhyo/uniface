# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


import cv2
import numpy as np

from uniface.attribute.base import Attribute
from uniface.constants import AgeGenderWeights
from uniface.face_utils import bbox_center_alignment
from uniface.log import Logger
from uniface.model_store import verify_model_weights
from uniface.onnx_utils import create_onnx_session
from uniface.types import AttributeResult

__all__ = ['AgeGender']


class AgeGender(Attribute):
    """
    Age and gender prediction model using ONNX Runtime.

    This class inherits from the base `Attribute` class and implements the
    functionality for predicting age (in years) and gender ID (0 for Female,
    1 for Male) from a face image. It requires a bounding box to locate the face.

    Args:
        model_name (AgeGenderWeights): The enum specifying the model weights to load.
            Defaults to `AgeGenderWeights.DEFAULT`.
        input_size (Optional[Tuple[int, int]]): Input size (height, width).
            If None, automatically detected from model metadata. Defaults to None.
    """

    def __init__(
        self,
        model_name: AgeGenderWeights = AgeGenderWeights.DEFAULT,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        """
        Initializes the AgeGender prediction model.

        Args:
            model_name (AgeGenderWeights): The enum specifying the model weights to load.
            input_size (Optional[Tuple[int, int]]): Input size (height, width).
                If None, automatically detected from model metadata. Defaults to None.
        """
        Logger.info(f'Initializing AgeGender with model={model_name.name}')
        self.model_path = verify_model_weights(model_name)
        self._user_input_size = input_size  # Store user preference
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

            # Use user-provided size if given, otherwise auto-detect from model
            model_input_size = tuple(input_meta.shape[2:4])  # (height, width)
            if self._user_input_size is not None:
                self.input_size = self._user_input_size
                if self._user_input_size != model_input_size:
                    Logger.warning(
                        f'Using custom input_size {self.input_size}, '
                        f'but model expects {model_input_size}. This may affect accuracy.'
                    )
            else:
                self.input_size = model_input_size

            self.output_names = [output.name for output in self.session.get_outputs()]
            Logger.info(f'Successfully initialized AgeGender model with input size {self.input_size}')
        except Exception as e:
            Logger.error(
                f"Failed to load AgeGender model from '{self.model_path}'",
                exc_info=True,
            )
            raise RuntimeError(f'Failed to initialize AgeGender model: {e}') from e

    def preprocess(self, image: np.ndarray, bbox: list | np.ndarray) -> np.ndarray:
        """
        Aligns the face based on the bounding box and preprocesses it for inference.

        Args:
            image (np.ndarray): The full input image in BGR format.
            bbox (Union[List, np.ndarray]): The face bounding box coordinates [x1, y1, x2, y2].

        Returns:
            np.ndarray: The preprocessed image blob ready for inference.
        """
        bbox = np.asarray(bbox)

        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        scale = self.input_size[1] / (max(width, height) * 1.5)

        # **Rotation parameter restored here**
        rotation = 0.0
        aligned_face, _ = bbox_center_alignment(image, center, self.input_size[1], scale, rotation)

        blob = cv2.dnn.blobFromImage(
            aligned_face,
            scalefactor=1.0,
            size=self.input_size[::-1],
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
        )
        return blob

    def postprocess(self, prediction: np.ndarray) -> AttributeResult:
        """
        Processes the raw model output to extract gender and age.

        Args:
            prediction (np.ndarray): The raw output from the model inference.

        Returns:
            AttributeResult: Result containing gender (0=Female, 1=Male) and age (in years).
        """
        # First two values are gender logits
        gender = int(np.argmax(prediction[:2]))
        # Third value is normalized age, scaled by 100
        age = int(np.round(prediction[2] * 100))
        return AttributeResult(gender=gender, age=age)

    def predict(self, image: np.ndarray, bbox: list | np.ndarray) -> AttributeResult:
        """
        Predicts age and gender for a single face specified by a bounding box.

        Args:
            image (np.ndarray): The full input image in BGR format.
            bbox (Union[List, np.ndarray]): The face bounding box coordinates [x1, y1, x2, y2].

        Returns:
            AttributeResult: Result containing gender (0=Female, 1=Male) and age (in years).
        """
        face_blob = self.preprocess(image, bbox)
        prediction = self.session.run(self.output_names, {self.input_name: face_blob})[0][0]
        return self.postprocess(prediction)
