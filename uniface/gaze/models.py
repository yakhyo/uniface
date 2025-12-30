# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


import cv2
import numpy as np

from uniface.constants import GazeWeights
from uniface.log import Logger
from uniface.model_store import verify_model_weights
from uniface.onnx_utils import create_onnx_session
from uniface.types import GazeResult

from .base import BaseGazeEstimator

__all__ = ['MobileGaze']


class MobileGaze(BaseGazeEstimator):
    """
    MobileGaze: Real-Time Gaze Estimation with ONNX Runtime.

    MobileGaze is a gaze estimation model that predicts gaze direction from a single
    face image. It supports multiple backbone architectures including ResNet 18/34/50,
    MobileNetV2, and MobileOne S0. The model uses a classification approach with binned
    angles, which are then decoded to continuous pitch and yaw values.

    The model outputs gaze direction as pitch (vertical) and yaw (horizontal) angles
    in radians.

    Reference:
        https://github.com/yakhyo/gaze-estimation

    Args:
        model_name (GazeWeights): The enum specifying the gaze model backbone to load.
            Options: RESNET18, RESNET34, RESNET50, MOBILENET_V2, MOBILEONE_S0.
            Defaults to `GazeWeights.RESNET18`.
        input_size (Tuple[int, int]): The resolution (width, height) for the model's
            input. Defaults to (448, 448).

    Attributes:
        input_size (Tuple[int, int]): Model input dimensions.
        input_mean (list): Per-channel mean values for normalization (ImageNet).
        input_std (list): Per-channel std values for normalization (ImageNet).

    Example:
        >>> from uniface.gaze import MobileGaze
        >>> from uniface import RetinaFace
        >>>
        >>> detector = RetinaFace()
        >>> gaze_estimator = MobileGaze()
        >>>
        >>> # Detect faces and estimate gaze for each
        >>> faces = detector.detect(image)
        >>> for face in faces:
        ...     bbox = face.bbox
        ...     x1, y1, x2, y2 = map(int, bbox[:4])
        ...     face_crop = image[y1:y2, x1:x2]
        ...     result = gaze_estimator.estimate(face_crop)
        ...     print(f'Gaze: pitch={np.degrees(result.pitch):.1f}°, yaw={np.degrees(result.yaw):.1f}°')
    """

    def __init__(
        self,
        model_name: GazeWeights = GazeWeights.RESNET34,
        input_size: tuple[int, int] = (448, 448),
    ) -> None:
        Logger.info(f'Initializing MobileGaze with model={model_name}, input_size={input_size}')

        self.input_size = input_size
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

        # Model specific parameters for bin-based classification (Gaze360 config)
        self._bins = 90
        self._binwidth = 4
        self._angle_offset = 180
        self._idx_tensor = np.arange(self._bins, dtype=np.float32)

        self.model_path = verify_model_weights(model_name)
        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Initialize the ONNX model from the stored model path.

        Raises:
            RuntimeError: If the model fails to load or initialize.
        """
        try:
            self.session = create_onnx_session(self.model_path)

            # Get input configuration
            input_cfg = self.session.get_inputs()[0]
            input_shape = input_cfg.shape
            self.input_name = input_cfg.name
            self.input_size = tuple(input_shape[2:4][::-1])  # Update from model

            # Get output configuration
            outputs = self.session.get_outputs()
            self.output_names = [output.name for output in outputs]

            if len(self.output_names) != 2:
                raise ValueError(f'Expected 2 output nodes (pitch, yaw), got {len(self.output_names)}')

            Logger.info(f'MobileGaze initialized with input size {self.input_size}')

        except Exception as e:
            Logger.error(f"Failed to load gaze model from '{self.model_path}'", exc_info=True)
            raise RuntimeError(f'Failed to initialize gaze model: {e}') from e

    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess a face crop for gaze estimation.

        Args:
            face_image (np.ndarray): A cropped face image in BGR format.

        Returns:
            np.ndarray: Preprocessed image tensor with shape (1, 3, H, W).
        """
        # Convert BGR to RGB
        image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        image = cv2.resize(image, self.input_size)

        # Normalize to [0, 1] and apply normalization
        image = image.astype(np.float32) / 255.0
        mean = np.array(self.input_mean, dtype=np.float32)
        std = np.array(self.input_std, dtype=np.float32)
        image = (image - mean) / std

        # HWC -> CHW -> NCHW
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0).astype(np.float32)

        return image

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax along axis 1."""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def postprocess(self, outputs: tuple[np.ndarray, np.ndarray]) -> GazeResult:
        """
        Postprocess raw model outputs into gaze angles.

        This method takes the raw output from the model's inference and
        converts it into pitch and yaw angles in radians.

        Args:
            outputs: Raw outputs from the model inference. The format depends
                     on the specific model architecture.

        Returns:
            GazeResult: Result containing pitch and yaw angles in radians.
        """
        pitch_logits, yaw_logits = outputs

        # Convert logits to probabilities
        pitch_probs = self._softmax(pitch_logits)
        yaw_probs = self._softmax(yaw_logits)

        # Compute expected bin index (soft-argmax)
        pitch_deg = np.sum(pitch_probs * self._idx_tensor, axis=1) * self._binwidth - self._angle_offset
        yaw_deg = np.sum(yaw_probs * self._idx_tensor, axis=1) * self._binwidth - self._angle_offset

        # Convert degrees to radians
        pitch = float(np.radians(pitch_deg[0]))
        yaw = float(np.radians(yaw_deg[0]))

        return GazeResult(pitch=pitch, yaw=yaw)

    def estimate(self, face_image: np.ndarray) -> GazeResult:
        """
        Perform end-to-end gaze estimation on a face image.

        This method orchestrates the full pipeline: preprocessing the input,
        running inference, and postprocessing to return the gaze direction.
        """
        input_tensor = self.preprocess(face_image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        return self.postprocess((outputs[0], outputs[1]))
