import cv2
import numpy as np
import onnxruntime
from typing import Tuple

from uniface.log import Logger
from uniface.face_utils import bbox_center_alignment
from uniface.model_store import verify_model_weights
from uniface.constants import AgeGenderWeights

__all__ = ["AgeGender"]


class AgeGender:
    """
    Age and Gender Prediction Model.
    """
    def __init__(self, model_name: AgeGenderWeights = AgeGenderWeights.DEFAULT, input_size:Tuple[int, int] = (112, 112)) -> None:
        """
        Initializes the Attribute model for inference.
        
        Args:
            model_path (str): Path to the ONNX file.
        """
        
        Logger.info(
            f"Initializing RetinaFace with model={model_name}, conf_thresh={conf_thresh}, nms_thresh={nms_thresh}, "
            f"pre_nms_topk={pre_nms_topk}, post_nms_topk={post_nms_topk}, dynamic_size={dynamic_size}, "
            f"input_size={input_size}"
        )
        
        self.model_path = model_path

        self.input_std = 1.0
        self.input_mean = 0.0

        self._initialize_model(model_path=model_path)

    def _initialize_model(self, model_path: str):
        """Initialize the model from the given path.

        Args:
            model_path (str): Path to .onnx model.
        """
        try:
            self.session = onnxruntime.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]

            )

            # Get model info
            metadata = self.session.get_inputs()[0]
            input_shape = metadata.shape
            self.input_size = tuple(input_shape[2:4][::-1])

            self.input_names = [x.name for x in self.session.get_inputs()]
            self.output_names = [x.name for x in self.session.get_outputs()]

        except Exception as e:
            print(f"Failed to load the model: {e}")
            raise

    def preprocess(self, image: np.ndarray, bbox: np.ndarray):
        """Preprocessing

        Args:
            image (np.ndarray): Numpy image
            bbox (np.ndarray): Bounding box coordinates: [x1, y1, x2, y2]

        Returns:
            np.ndarray: Transformed image
        """
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        scale = self.input_size[0] / (max(width, height)*1.5)
        rotation = 0.0

        transformed_image, M = bbox_center_alignment(image, center, self.input_size[0], scale, rotation)

        input_size = tuple(transformed_image.shape[0:2][::-1])

        blob = cv2.dnn.blobFromImage(
            transformed_image,
            1.0/self.input_std,
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True
        )
        return blob

    def postprocess(self, predictions: np.ndarray) -> Tuple[np.int64, int]:
        """Postprocessing

        Args:
            predictions (np.ndarray): Model predictions, shape: [1, 3]

        Returns:
            Tuple[np.int64, int]: Gender and Age values
        """
        gender = np.argmax(predictions[:2])
        age = int(np.round(predictions[2]*100))
        return gender, age

    def get(self, image: np.ndarray, bbox: np.ndarray) -> Tuple[np.int64, int]:
        blob = self.preprocess(image, bbox)
        predictions = self.session.run(self.output_names, {self.input_names[0]: blob})[0][0]
        gender, age = self.postprocess(predictions)

        return gender, age
