import cv2
import numpy as np
import onnxruntime
from typing import Tuple

from uniface.log import Logger
from uniface.face_utils import bbox_center_alignment
from uniface.model_store import verify_model_weights
from uniface.constants import AgeGenderWeights

from uniface.detection import RetinaFace
from uniface.constants import RetinaFaceWeights

__all__ = ["AgeGender"]


class AgeGender:
    """
    Age and Gender Prediction Model.
    """

    def __init__(self, model_name: AgeGenderWeights = AgeGenderWeights.DEFAULT, input_size: Tuple[int, int] = (112, 112)) -> None:
        """
        Initializes the Attribute model for inference.

        Args:
            model_path (str): Path to the ONNX file.
        """

        Logger.info(
            f"Initializing AgeGender with model={model_name}, "
            f"input_size={input_size}"
        )

        self.input_size = input_size
        self.input_std = 1.0
        self.input_mean = 0.0

        # Get path to model weights
        self._model_path = verify_model_weights(model_name)
        Logger.info(f"Verfied model weights located at: {self._model_path}")

        # Initialize model
        self._initialize_model(model_path=self._model_path)

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
        scale = self.input_size[0] / (max(width, height) * 1.5)
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

    def predict(self, image: np.ndarray, bbox: np.ndarray) -> Tuple[np.int64, int]:
        blob = self.preprocess(image, bbox)
        predictions = self.session.run(self.output_names, {self.input_names[0]: blob})[0][0]
        gender, age = self.postprocess(predictions)

        return gender, age


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
    age_detector = AgeGender()

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

            gender, age = age_detector.predict(frame, box[:4])

            txt = f"{gender} ({age:.2f})"
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
