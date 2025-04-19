# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import cv2
import numpy as np
import torch
from PIL import Image

from typing import Tuple, Union

from uniface.log import Logger
from uniface import RetinaFace
from uniface.face_utils import face_alignment
from uniface.model_store import verify_model_weights
from uniface.constants import RetinaFaceWeights, DDAMFNWeights


class Emotion:
    """
    Emotion recognition using a TorchScript model.

    Args:
        model_name (DDAMFNWeights): Pretrained model enum. Defaults to AFFECNET7.

    Attributes:
        emotions (List[str]): Emotion label list.
        device (torch.device): Inference device (CPU or CUDA).
        model (torch.jit.ScriptModule): Loaded TorchScript model.

    Raises:
        ValueError: If model weights are invalid or not found.
        RuntimeError: If model loading fails.
    """

    def __init__(self, model_name: DDAMFNWeights = DDAMFNWeights.AFFECNET7, input_size: Tuple[int, int] = (112, 112)) -> None:
        """
        Initialize the emotion detector with a TorchScript model
        """
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.emotions = [
            "Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Angry"
        ]
        if model_name == DDAMFNWeights.AFFECNET8:
            self.emotions.append("Contempt")

        self.input_size = input_size
        
        Logger.info(
            f"Initialized Emotion class with model={model_name.name}, "
            f"device={'cuda' if torch.cuda.is_available() else 'cpu'}, "
            f"num_classes={len(self.emotions)}, input_size={self.input_size}"
        )

        # Get path to model weights
        self._model_path = verify_model_weights(model_name)
        Logger.info(f"Verified model weights located at: {self._model_path}")

        # Initialize model
        self._initialize_model(model_path=self._model_path)

    def _initialize_model(self, model_path: str) -> None:
        """
        Initializes a TorchScript model for emotion inference.

        Args:
            model_path (str): Path to the TorchScript (.pt) model.
        """
        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            Logger.info(f"TorchScript model successfully loaded from: {model_path}")

            # Warm-up
            dummy = torch.randn(1, 3, 112, 112).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy)
            Logger.info("Emotion model warmed up with dummy input.")

        except Exception as e:
            Logger.error(f"Failed to load TorchScript model from {model_path}: {e}")
            raise

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Resize, normalize and convert image to tensor manually without torchvision.

        Args:
            image (np.ndarray): BGR image (H, W, 3)
        Returns:
            torch.Tensor: Preprocessed image tensor of shape (1, 3, 112, 112)
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR -> RGB
        
        # Resize to (112, 112)
        image = cv2.resize(image, self.input_size).astype(np.float32) / 255.0

        # Normalize with mean and std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_normalized = (image - mean) / std

        # HWC to CHW
        image_transposed = image_normalized.transpose((2, 0, 1))

        # Convert to torch tensor and add batch dimension
        tensor = torch.from_numpy(image_transposed).unsqueeze(0).to(self.device)

        return tensor

    def predict(self, image: np.ndarray, landmark: np.ndarray) -> Tuple[Union[str, None], Union[float, None]]:
        """
        Predict the emotion from an BGR face image.

        Args:
            image (np.ndarray): Input face image in RGB format.
            landmark (np.ndarray): Facial five point landmark.

        Returns:
            Tuple[str, float]: (Predicted emotion label, Confidence score)

        Raises:
            RuntimeError: If the input is invalid or inference fails internally.
        """
        if not isinstance(image, np.ndarray):
            Logger.error("Input must be a NumPy ndarray.")
            raise ValueError("Input must be a NumPy ndarray (RGB image).")

        if image.ndim != 3 or image.shape[2] != 3:
            Logger.error(f"Invalid image shape: {image.shape}. Expected HxWx3 RGB image.")
            raise ValueError("Input image must be in RGB format with shape (H, W, 3).")

        try:
            image, _ = face_alignment(image, landmark)
            tensor = self.preprocess(image)

            with torch.no_grad():
                output = self.model(tensor)

                if isinstance(output, tuple):
                    output = output[0]

                probs = torch.nn.functional.softmax(output, dim=1).squeeze(0).cpu().numpy()
                pred_idx = int(np.argmax(probs))
                confidence = round(float(probs[pred_idx]), 2)

                return self.emotions[pred_idx], confidence

        except Exception as e:
            Logger.error(f"Emotion inference failed: {e}")
            return None, None


# TODO: For testing purposes only, remove later

def main():

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
