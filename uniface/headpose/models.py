# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import cv2
import numpy as np

from uniface.constants import HeadPoseWeights
from uniface.log import Logger
from uniface.model_store import verify_model_weights
from uniface.onnx_utils import create_onnx_session
from uniface.types import HeadPoseResult

from .base import BaseHeadPoseEstimator

__all__ = ['HeadPose']


class HeadPose(BaseHeadPoseEstimator):
    """
    Head Pose Estimation with ONNX Runtime using 6D Rotation Representation.

    This model estimates head orientation from a single face image by predicting
    a 3x3 rotation matrix (via continuous 6D representation) and converting it to
    Euler angles (pitch, yaw, roll) in degrees.

    Supports multiple backbone architectures: ResNet-18/34/50, MobileNetV2,
    and MobileNetV3 (small/large).

    Reference:
        https://github.com/yakhyo/head-pose-estimation

    Args:
        model_name (HeadPoseWeights): The enum specifying the head pose model to load.
            Options: RESNET18, RESNET34, RESNET50, MOBILENET_V2, MOBILENET_V3_SMALL,
            MOBILENET_V3_LARGE. Defaults to `HeadPoseWeights.RESNET18`.
        input_size (tuple[int, int]): The resolution (width, height) for the model's
            input. Defaults to (224, 224).
        providers (list[str] | None): ONNX Runtime execution providers. If None, auto-detects
            the best available provider. Example: ['CPUExecutionProvider'] to force CPU.

    Attributes:
        input_size (tuple[int, int]): Model input dimensions.
        input_mean (np.ndarray): Per-channel mean values for normalization (ImageNet).
        input_std (np.ndarray): Per-channel std values for normalization (ImageNet).

    Example:
        >>> from uniface.headpose import HeadPose
        >>> from uniface import RetinaFace
        >>>
        >>> detector = RetinaFace()
        >>> head_pose = HeadPose()
        >>>
        >>> faces = detector.detect(image)
        >>> for face in faces:
        ...     bbox = face.bbox
        ...     x1, y1, x2, y2 = map(int, bbox[:4])
        ...     face_crop = image[y1:y2, x1:x2]
        ...     result = head_pose.estimate(face_crop)
        ...     print(f'Pose: pitch={result.pitch:.1f}°, yaw={result.yaw:.1f}°, roll={result.roll:.1f}°')
    """

    def __init__(
        self,
        model_name: HeadPoseWeights = HeadPoseWeights.RESNET18,
        input_size: tuple[int, int] = (224, 224),
        providers: list[str] | None = None,
    ) -> None:
        Logger.info(f'Initializing HeadPose with model={model_name}, input_size={input_size}')

        self.input_size = input_size
        self.input_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.input_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.providers = providers

        self.model_path = verify_model_weights(model_name)
        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Initialize the ONNX model from the stored model path.

        Raises:
            RuntimeError: If the model fails to load or initialize.
        """
        try:
            self.session = create_onnx_session(self.model_path, providers=self.providers)

            input_cfg = self.session.get_inputs()[0]
            input_shape = input_cfg.shape
            self.input_name = input_cfg.name
            self.input_size = tuple(input_shape[2:4][::-1])

            outputs = self.session.get_outputs()
            self.output_names = [output.name for output in outputs]

            if len(self.output_names) != 1:
                raise ValueError(f'Expected 1 output node (rotation_matrix), got {len(self.output_names)}')

            Logger.info(f'HeadPose initialized with input size {self.input_size}')

        except Exception as e:
            Logger.error(f"Failed to load head pose model from '{self.model_path}'", exc_info=True)
            raise RuntimeError(f'Failed to initialize head pose model: {e}') from e

    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess a face crop for head pose estimation.

        Args:
            face_image (np.ndarray): A cropped face image in BGR format.

        Returns:
            np.ndarray: Preprocessed image tensor with shape (1, 3, H, W).
        """
        image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_size)
        image = image.astype(np.float32) / 255.0
        image = (image - self.input_mean) / self.input_std

        # HWC -> CHW -> NCHW
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0).astype(np.float32)

        return image

    @staticmethod
    def rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> np.ndarray:
        """Convert (B, 3, 3) rotation matrices to Euler angles in degrees.

        Uses the ZYX convention to decompose rotation matrices into
        pitch (X), yaw (Y), and roll (Z) angles.

        Args:
            rotation_matrix: Batch of rotation matrices with shape (B, 3, 3).

        Returns:
            np.ndarray: Euler angles with shape (B, 3) as [pitch, yaw, roll] in degrees.
        """
        R = rotation_matrix
        sy = np.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
        singular = sy < 1e-6

        x = np.where(singular, np.arctan2(-R[:, 1, 2], R[:, 1, 1]), np.arctan2(R[:, 2, 1], R[:, 2, 2]))
        y = np.arctan2(-R[:, 2, 0], sy)
        z = np.where(singular, np.zeros_like(sy), np.arctan2(R[:, 1, 0], R[:, 0, 0]))

        return np.degrees(np.stack([x, y, z], axis=1))

    def postprocess(self, rotation_matrix: np.ndarray) -> HeadPoseResult:
        """
        Convert a rotation matrix into Euler angles.

        Args:
            rotation_matrix: Rotation matrix with shape (B, 3, 3).

        Returns:
            HeadPoseResult: Result containing pitch, yaw, and roll in degrees.
        """
        euler = self.rotation_matrix_to_euler(rotation_matrix)
        return HeadPoseResult(
            pitch=float(euler[0, 0]),
            yaw=float(euler[0, 1]),
            roll=float(euler[0, 2]),
        )

    def estimate(self, face_image: np.ndarray) -> HeadPoseResult:
        """
        Perform end-to-end head pose estimation on a face image.

        This method orchestrates the full pipeline: preprocessing the input,
        running inference, and postprocessing to return the head orientation.
        """
        input_tensor = self.preprocess(face_image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        rotation_matrix = outputs[0]  # (1, 3, 3)

        return self.postprocess(rotation_matrix)
