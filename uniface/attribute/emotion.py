# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import cv2
import torch
import numpy as np
from typing import Tuple, Union, List

from uniface.attribute.base import Attribute
from uniface.log import Logger
from uniface.constants import DDAMFNWeights
from uniface.face_utils import face_alignment
from uniface.model_store import verify_model_weights

__all__ = ["Emotion"]


class Emotion(Attribute):
    """
    Emotion recognition model using a TorchScript model.

    This class inherits from the base `Attribute` class and implements the
    functionality for predicting one of several emotion categories from a face
    image. It requires 5-point facial landmarks for alignment.
    """

    def __init__(
        self,
        model_weights: DDAMFNWeights = DDAMFNWeights.AFFECNET7,
        input_size: Tuple[int, int] = (112, 112),
    ) -> None:
        """
        Initializes the emotion recognition model.

        Args:
            model_weights (DDAMFNWeights): The enum for the model weights to load.
            input_size (Tuple[int, int]): The expected input size for the model.
        """
        Logger.info(f"Initializing Emotion with model={model_weights.name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.model_path = verify_model_weights(model_weights)

        # Define emotion labels based on the selected model
        self.emotion_labels = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Angry"]
        if model_weights == DDAMFNWeights.AFFECNET8:
            self.emotion_labels.append("Contempt")

        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Loads and initializes the TorchScript model for inference.
        """
        try:
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
            # Warm-up with a dummy input for faster first inference
            dummy_input = torch.randn(1, 3, *self.input_size).to(self.device)
            with torch.no_grad():
                self.model(dummy_input)
            Logger.info(f"Successfully initialized Emotion model on {self.device}")
        except Exception as e:
            Logger.error(f"Failed to load Emotion model from '{self.model_path}'", exc_info=True)
            raise RuntimeError(f"Failed to initialize Emotion model: {e}")

    def preprocess(self, image: np.ndarray, landmark: Union[List, np.ndarray]) -> torch.Tensor:
        """
        Aligns the face using landmarks and preprocesses it into a tensor.

        Args:
            image (np.ndarray): The full input image in BGR format.
            landmark (Union[List, np.ndarray]): The 5-point facial landmarks.

        Returns:
            torch.Tensor: The preprocessed image tensor ready for inference.
        """
        landmark = np.asarray(landmark)
        
        aligned_image, _ = face_alignment(image, landmark)

        # Convert BGR to RGB, resize, normalize, and convert to a CHW tensor
        rgb_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(rgb_image, self.input_size).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized_image = (resized_image - mean) / std
        transposed_image = normalized_image.transpose((2, 0, 1))

        return torch.from_numpy(transposed_image).unsqueeze(0).to(self.device)

    def postprocess(self, prediction: torch.Tensor) -> Tuple[str, float]:
        """
        Processes the raw model output to get the emotion label and confidence score.
        """
        probabilities = torch.nn.functional.softmax(prediction, dim=1).squeeze().cpu().numpy()
        pred_index = np.argmax(probabilities)
        emotion_label = self.emotion_labels[pred_index]
        confidence = float(probabilities[pred_index])
        return emotion_label, confidence

    def predict(self, image: np.ndarray, landmark: Union[List, np.ndarray]) -> Tuple[str, float]:
        """
        Predicts the emotion from a single face specified by its landmarks.
        """
        input_tensor = self.preprocess(image, landmark)
        with torch.no_grad():
            output = self.model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]

        return self.postprocess(output)


# TODO: below is only for testing, remove it later
if __name__ == "__main__":
    from uniface.detection import create_detector
    from uniface.constants import RetinaFaceWeights

    print("Initializing models for live inference...")
    # 1. Initialize the face detector
    # Using a smaller model for faster real-time performance
    detector = create_detector(model_name=RetinaFaceWeights.MNET_V2)

    # 2. Initialize the attribute predictor
    emotion_predictor = Emotion()

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

        # Detect faces in the current frame.
        # This method returns a list of dictionaries for each detected face.
        detections = detector.detect(frame)

        # For each detected face, predict the emotion
        for detection in detections:
            box = detection['bbox']
            landmark = detection['landmarks']
            x1, y1, x2, y2 = map(int, box)

            # Predict attributes using the landmark
            emotion, confidence = emotion_predictor.predict(frame, landmark)
            
            # Prepare text and draw on the frame
            label = f"{emotion} ({confidence:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow("Emotion Inference (Press 'q' to quit)", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Inference stopped.")