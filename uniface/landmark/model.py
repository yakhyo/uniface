import cv2
import onnx
import onnxruntime as ort
import numpy as np

from typing import Tuple

from uniface.log import Logger
from uniface.face_utils import bbox_center_alignment, trans_points
from uniface.model_store import verify_model_weights

from uniface.detection import RetinaFace
from uniface.constants import RetinaFaceWeights, LandmarkWeights

__all__ = ['Landmark']


class Landmark:
    def __init__(self, model_name: LandmarkWeights = LandmarkWeights.DEFAULT, input_size: Tuple[int, int] = (192, 192)) -> None:
        """
        Initializes the Attribute model for inference.

        Args:
            model_path (str): Path to the ONNX file.
        """

        Logger.info(
            f"Initializing Landmark with model={model_name}, "
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

        
    def _initialize_model(self, model_path:str):
        """ Initialize the model from the given path.
        Args:
            model_path (str): Path to .onnx model.
        """
        try:
            self.session = ort.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )

            metadata = self.session.get_inputs()[0]
            input_shape = metadata.shape
            self.input_size = tuple(input_shape[2:4][::-1])

            self.input_names = [x.name for x in self.session.get_inputs()]
            self.output_names = [x.name for x in self.session.get_outputs()]

            outputs = self.session.get_outputs()
            output_shape = outputs[0].shape
            self.lmk_dim = 2
            self.lmk_num = output_shape[1] // self.lmk_dim

        except Exception as e:
            print(f"Failed to load the model: {e}")
            raise

    def preprocess(self, image: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the input image and bbox for inference.

        Args:
            image (np.ndarray): Input image.
            bbox (np.ndarray): Bounding box [x1, y1, x2, y2].

        Returns:
            Tuple[np.ndarray, np.ndarray]: Preprocessed blob and transformation matrix.
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
        return blob, M
    
    def postprocess(self, preds: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        Postprocess model outputs to get landmarks.

        Args:
            preds (np.ndarray): Raw model predictions.
            M (np.ndarray): Affine transformation matrix.

        Returns:
            np.ndarray: Transformed landmarks.
        """

        preds = preds.reshape((-1, 2))

        preds[:, 0:2] += 1
        preds[:, 0:2] *= (self.input_size[0] // 2)

        IM = cv2.invertAffineTransform(M)
        preds = trans_points(preds, IM)

        return preds
    
    def predict(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Predict facial landmarks for the given image and bounding box.

        Args:
            image (np.ndarray): Input image.
            bbox (np.ndarray): Bounding box [x1, y1, x2, y2].

        Returns:
            np.ndarray: Predicted landmarks.
        """
        blob, M = self.preprocess(image, bbox)
        preds = self.session.run(self.output_names, {self.input_names[0]: blob})[0][0]
        landmarks = self.postprocess(preds, M)

        return landmarks

# TODO: For testing purposes only, remote later

if __name__ == "__main__":

    face_detector = RetinaFace(
        model_name=RetinaFaceWeights.MNET_V2,
        conf_thresh=0.5,
        pre_nms_topk=5000,
        nms_thresh=0.4,
        post_nms_topk=750,
        dynamic_size=False,
        input_size=(640, 640)
    )

    model = Landmark()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not available.")
        exit()

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed.")
            break

        boxes, landmarks = face_detector.detect(frame)

        if boxes is None or len(boxes) == 0:
            cv2.imshow("Facial Landmark Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        for box in boxes:
            x1, y1, x2, y2, score = box.astype(int)

            lmk = model.predict(frame, box[:4])

            for (x, y) in lmk.astype(int):
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow("Facial Landmark Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
