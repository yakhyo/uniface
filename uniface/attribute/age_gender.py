# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple, Union, List

from uniface.attribute.base import Attribute
from uniface.log import Logger
from uniface.constants import AgeGenderWeights
from uniface.face_utils import bbox_center_alignment
from uniface.model_store import verify_model_weights

__all__ = ["AgeGender"]


class AgeGender(Attribute):
    """
    Age and gender prediction model using ONNX Runtime.

    This class inherits from the base `Attribute` class and implements the
    functionality for predicting age (in years) and gender (0 for female,
    1 for male) from a face image. It requires a bounding box to locate the face.
    """

    def __init__(self, model_name: AgeGenderWeights = AgeGenderWeights.DEFAULT) -> None:
        """
        Initializes the AgeGender prediction model.

        Args:
            model_name (AgeGenderWeights): The enum specifying the model weights
                                           to load.
        """
        Logger.info(f"Initializing AgeGender with model={model_name.name}")
        self.model_path = verify_model_weights(model_name)
        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Initializes the ONNX model and creates an inference session.
        """
        try:
            self.session = ort.InferenceSession(
                self.model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            # Get model input details from the loaded model
            input_meta = self.session.get_inputs()[0]
            self.input_name = input_meta.name
            self.input_size = tuple(input_meta.shape[2:4])  # (height, width)
            self.output_names = [output.name for output in self.session.get_outputs()]
            Logger.info(f"Successfully initialized AgeGender model with input size {self.input_size}")
        except Exception as e:
            Logger.error(f"Failed to load AgeGender model from '{self.model_path}'", exc_info=True)
            raise RuntimeError(f"Failed to initialize AgeGender model: {e}")

    def preprocess(self, image: np.ndarray, bbox: Union[List, np.ndarray]) -> np.ndarray:
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
        aligned_face, _ = bbox_center_alignment(
            image, center, self.input_size[1], scale, rotation
        )

        blob = cv2.dnn.blobFromImage(
            aligned_face,
            scalefactor=1.0,
            size=self.input_size[::-1],
            mean=(0.0, 0.0, 0.0),
            swapRB=True
        )
        return blob

    def postprocess(self, prediction: np.ndarray) -> Tuple[str, int]:
        """
        Processes the raw model output to extract gender and age.

        Args:
            prediction (np.ndarray): The raw output from the model inference.

        Returns:
            Tuple[str, int]: A tuple containing the predicted gender label ("Female" or "Male")
                             and age (in years).
        """
        # First two values are gender logits
        gender_id = int(np.argmax(prediction[:2]))
        gender = "Female" if gender_id == 0 else "Male"
        # Third value is normalized age, scaled by 100
        age = int(np.round(prediction[2] * 100))
        return gender, age

    def predict(self, image: np.ndarray, bbox: Union[List, np.ndarray]) -> Tuple[str, int]:
        """
        Predicts age and gender for a single face specified by a bounding box.

        Args:
            image (np.ndarray): The full input image in BGR format.
            bbox (Union[List, np.ndarray]): The face bounding box coordinates [x1, y1, x2, y2].

        Returns:
            Tuple[str, int]: A tuple containing the predicted gender label and age.
        """
        face_blob = self.preprocess(image, bbox)
        prediction = self.session.run(self.output_names, {self.input_name: face_blob})[0][0]
        gender, age = self.postprocess(prediction)
        return gender, age


# TODO: below is only for testing, remove it later
if __name__ == "__main__":
    # To run this script, you need to have uniface.detection installed
    # or available in your path.
    from uniface.detection import create_detector
    from uniface.constants import RetinaFaceWeights

    print("Initializing models for live inference...")
    # 1. Initialize the face detector
    # Using a smaller model for faster real-time performance
    detector = create_detector(model_name=RetinaFaceWeights.MNET_V2)

    # 2. Initialize the attribute predictor
    age_gender_predictor = AgeGender()

    # 3. Start webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    print("Starting webcam feed. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Detect faces in the current frame
        detections = detector.detect(frame)

        # For each detected face, predict age and gender
        for detection in detections:
            box = detection['bbox']
            x1, y1, x2, y2 = map(int, box)

            # Predict attributes
            gender, age = age_gender_predictor.predict(frame, box)

            # Prepare text and draw on the frame
            label = f"{gender}, {age}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow("Age and Gender Inference (Press 'q' to quit)", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Inference stopped.")
