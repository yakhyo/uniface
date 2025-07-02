# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import cv2
import torch
import numpy as np

from typing import Tuple, Union

from uniface.log import Logger
from uniface.constants import DDAMFNWeights
from uniface.face_utils import face_alignment
from uniface.model_store import verify_model_weights


class Emotion:
    """
    Emotion recognition using a TorchScript model.

    Args:
        model_weights (DDAMFNWeights): Pretrained model weights enum. Defaults to AFFECNET7.
        input_size (Tuple[int, int]): Size of input images. Defaults to (112, 112).

    Attributes:
        emotion_labels (List[str]): List of emotion labels the model can predict.
        device (torch.device): Inference device (CPU or CUDA).
        model (torch.jit.ScriptModule): Loaded TorchScript model.

    Raises:
        ValueError: If model weights are invalid or not found.
        RuntimeError: If model loading fails.
    """

    def __init__(
            self,
            model_weights: DDAMFNWeights = DDAMFNWeights.AFFECNET7,
            input_size: Tuple[int, int] = (112, 112)
    ) -> None:
        """
        Initialize the emotion detector with a TorchScript model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emotion_labels = [
            "Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Angry"
        ]

        # Add contempt for AFFECNET8 model
        if model_weights == DDAMFNWeights.AFFECNET8:
            self.emotion_labels.append("Contempt")

        # Initialize image preprocessing parameters
        self.input_size = input_size
        self.normalization_std = [0.229, 0.224, 0.225]
        self.normalization_mean = [0.485, 0.456, 0.406]

        Logger.info(
            f"Initialized Emotion class with model={model_weights.name}, "
            f"device={'cuda' if torch.cuda.is_available() else 'cpu'}, "
            f"num_classes={len(self.emotion_labels)}, input_size={self.input_size}"
        )

        # Get path to model weights and initialize model
        self.model_path = verify_model_weights(model_weights)
        Logger.info(f"Verified model weights located at: {self.model_path}")
        self._load_model()

    def _load_model(self) -> None:
        """
        Loads and initializes a TorchScript model for emotion inference.

        Raises:
            RuntimeError: If loading the model fails.
        """
        try:
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
            Logger.info(f"TorchScript model successfully loaded from: {self.model_path}")

            # Warm-up with dummy input
            dummy_input = torch.randn(1, 3, *self.input_size).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
            Logger.info("Emotion model warmed up with dummy input.")

        except Exception as e:
            Logger.error(f"Failed to load TorchScript model from {self.model_path}: {e}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model inference: resize, normalize and convert to tensor.

        Args:
            image (np.ndarray): BGR image (H, W, 3)

        Returns:
            torch.Tensor: Preprocessed image tensor of shape (1, 3, H, W)
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to target input size
        resized_image = cv2.resize(rgb_image, self.input_size).astype(np.float32) / 255.0

        # Normalize with mean and std
        mean_array = np.array(self.normalization_mean, dtype=np.float32)
        std_array = np.array(self.normalization_std, dtype=np.float32)
        normalized_image = (resized_image - mean_array) / std_array

        # Convert from HWC to CHW format
        transposed_image = normalized_image.transpose((2, 0, 1))

        # Convert to torch tensor and add batch dimension
        tensor = torch.from_numpy(transposed_image).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, image: np.ndarray, landmark: np.ndarray) -> Tuple[Union[str, None], Union[float, None]]:
        """
        Predict the emotion from a face image.

        Args:
            image (np.ndarray): Input face image in BGR format.
            landmark (np.ndarray): Facial five point landmark.

        Returns:
            Tuple[str, float]: (Predicted emotion label, Confidence score)
            Returns (None, None) if prediction fails.

        Raises:
            ValueError: If the input is not a valid BGR image.
        """
        # Validate input
        if not isinstance(image, np.ndarray):
            Logger.error("Input must be a NumPy ndarray.")
            raise ValueError("Input must be a NumPy ndarray (BGR image).")

        if image.ndim != 3 or image.shape[2] != 3:
            Logger.error(f"Invalid image shape: {image.shape}. Expected HxWx3 image.")
            raise ValueError("Input image must have shape (H, W, 3).")

        try:
            # Align face using landmarks
            aligned_image, _ = face_alignment(image, landmark)

            # Preprocess and run inference
            input_tensor = self.preprocess(aligned_image)

            with torch.no_grad():
                output = self.model(input_tensor)

                # Handle case where model returns a tuple
                if isinstance(output, tuple):
                    output = output[0]

                # Get probabilities and prediction
                probabilities = torch.nn.functional.softmax(output, dim=1).squeeze(0).cpu().numpy()
                predicted_index = int(np.argmax(probabilities))
                confidence_score = round(float(probabilities[predicted_index]), 2)

                return self.emotion_labels[predicted_index], confidence_score

        except Exception as e:
            Logger.error(f"Emotion inference failed: {e}")
            return None, None


# TODO: For testing purposes only, remove later

def main():
    from uniface import RetinaFace
    from uniface.constants import RetinaFaceWeights

    face_detector = RetinaFace(
        model_name=RetinaFaceWeights.MNET_V2,
        conf_thresh=0.5,
        pre_nms_topk=5000,
        nms_thresh=0.4,
        post_nms_topk=750,
        dynamic_size=False,
        input_size=(640, 640)
    )
    emotion_detector = Emotion()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not available.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed.")
            break

        boxes, landmarks = face_detector.detect(frame)

        for box, landmark in zip(boxes, landmarks):
            x1, y1, x2, y2, score = box.astype(int)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            emotion, preds = emotion_detector.predict(frame, landmark)

            txt = f"{emotion} ({preds:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, txt, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Face + Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
