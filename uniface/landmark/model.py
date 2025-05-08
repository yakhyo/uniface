import cv2
import onnx
import onnxruntime as ort
import numpy as np

from typing import Tuple

from uniface.log import Logger
from uniface.constants import LandmarkWeights
from uniface.model_store import verify_model_weights
from uniface.face_utils import bbox_center_alignment, transform_points_2d

__all__ = ['Landmark']


class Landmark:
    """
    Facial landmark detection model for predicting facial keypoints.
    """
    
    def __init__(
        self, 
        model_name: LandmarkWeights = LandmarkWeights.DEFAULT, 
        input_size: Tuple[int, int] = (192, 192)
    ) -> None:
        """
        Initializes the Facial Landmark model for inference.

        Args:
            model_name: Enum specifying which landmark model weights to use
            input_size: Input resolution for the model (width, height)
        """
        Logger.info(
            f"Initializing Facial Landmark with model={model_name}, "
            f"input_size={input_size}"
        )

        # Initialize configuration
        self.input_size = input_size
        self.input_std = 1.0
        self.input_mean = 0.0

        # Get path to model weights
        self.model_path = verify_model_weights(model_name)
        Logger.info(f"Verified model weights located at: {self.model_path}")

        # Initialize model
        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize the ONNX model from the stored model path.
        
        Raises:
            RuntimeError: If the model fails to load or initialize.
        """
        try:
            self.session = ort.InferenceSession(
                self.model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )

            # Get input configuration
            input_metadata = self.session.get_inputs()[0]
            input_shape = input_metadata.shape
            self.input_size = tuple(input_shape[2:4][::-1])  # Update input size from model

            # Get input/output names
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]

            # Determine landmark dimensions from output shape
            output_shape = self.session.get_outputs()[0].shape
            self.lmk_dim = 2  # x,y coordinates
            self.lmk_num = output_shape[1] // self.lmk_dim  # Number of landmarks
            
            Logger.info(f"Model initialized with {self.lmk_num} landmarks")

        except Exception as e:
            Logger.error(f"Failed to load landmark model from '{self.model_path}'", exc_info=True)
            raise RuntimeError(f"Failed to initialize landmark model: {e}")

    def preprocess(self, image: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the input image and bounding box for inference.

        Args:
            image: Input image in BGR format
            bbox: Bounding box coordinates [x1, y1, x2, y2]

        Returns:
            Tuple containing:
                - Preprocessed image blob ready for inference
                - Transformation matrix for mapping predictions back to original image
        """
        # Calculate face dimensions and center
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        
        # Determine scale to fit face with some margin
        scale = self.input_size[0] / (max(width, height) * 1.5)
        rotation = 0.0

        # Align face using center, scale and rotation
        aligned_face, transform_matrix = bbox_center_alignment(
            image, center, self.input_size[0], scale, rotation
        )
        
        # Convert to blob format for inference
        face_blob = cv2.dnn.blobFromImage(
            aligned_face,
            1.0 / self.input_std,
            self.input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True  # Convert BGR to RGB
        )
        
        return face_blob, transform_matrix

    def postprocess(self, predictions: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """
        Convert raw model predictions to image coordinates.

        Args:
            predictions: Raw landmark coordinates from model output
            transform_matrix: Affine transformation matrix from preprocessing

        Returns:
            Landmarks in original image coordinates
        """
        # Reshape to pairs of x,y coordinates
        landmarks = predictions.reshape((-1, 2))

        # Denormalize coordinates to pixel space
        landmarks[:, 0:2] += 1  # Shift from [-1,1] to [0,2] range
        landmarks[:, 0:2] *= (self.input_size[0] // 2)  # Scale to pixel coordinates

        # Invert the transformation to map back to original image
        inverse_matrix = cv2.invertAffineTransform(transform_matrix)
        landmarks = transform_points_2d(landmarks, inverse_matrix)

        return landmarks

    def predict(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Predict facial landmarks for the given image and face bounding box.

        Args:
            image: Input image in BGR format
            bbox: Face bounding box [x1, y1, x2, y2]

        Returns:
            Array of facial landmarks in original image coordinates
        """
        # Preprocess image
        face_blob, transform_matrix = self.preprocess(image, bbox)
        
        # Run inference
        raw_predictions = self.session.run(
            self.output_names, 
            {self.input_names[0]: face_blob}
        )[0][0]
        
        # Postprocess to get landmarks in original image space
        landmarks = self.postprocess(raw_predictions, transform_matrix)

        return landmarks

# TODO: For testing purposes only, remote later


if __name__ == "__main__":
    from uniface.detection import RetinaFace
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
