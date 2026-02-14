# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import cv2
import numpy as np

from uniface.constants import XSegWeights
from uniface.face_utils import face_alignment
from uniface.log import Logger
from uniface.model_store import verify_model_weights
from uniface.onnx_utils import create_onnx_session

from .base import BaseFaceParser

__all__ = ['XSeg']


class XSeg(BaseFaceParser):
    """
    XSeg: Face Segmentation Model from DeepFaceLab with ONNX Runtime.

    XSeg outputs a mask for face regions. Unlike BiSeNet which works
    on bbox crops, XSeg requires 5-point landmarks for face alignment. The model
    uses NHWC input format and outputs values in [0, 1] range.

    Reference:
        https://github.com/iperov/DeepFaceLab

    Args:
        model_name (XSegWeights): The enum specifying the XSeg model to load.
            Defaults to `XSegWeights.DEFAULT`.
        align_size (int): Face alignment output size. Must be multiple of 112 or 128.
            Defaults to 256.
        blur_sigma (float): Gaussian blur sigma for mask smoothing.
            0 = raw output (no blur). Defaults to 0.
        providers (list[str] | None): ONNX Runtime execution providers. If None,
            auto-detects the best available provider.

    Attributes:
        align_size (int): Face alignment output size.
        blur_sigma (float): Blur sigma for post-processing.
        input_size (tuple[int, int]): Model input dimensions (width, height).

    Example:
        >>> from uniface.parsing import XSeg
        >>> from uniface import RetinaFace
        >>>
        >>> detector = RetinaFace()
        >>> parser = XSeg()
        >>>
        >>> faces = detector.detect(image)
        >>> for face in faces:
        ...     if face.landmarks is not None:
        ...         mask = parser.parse(image, landmarks=face.landmarks)
        ...         print(f'Mask shape: {mask.shape}')
    """

    def __init__(
        self,
        model_name: XSegWeights = XSegWeights.DEFAULT,
        align_size: int = 256,
        blur_sigma: float = 0,
        providers: list[str] | None = None,
    ) -> None:
        Logger.info(f'Initializing XSeg with model={model_name}, align_size={align_size}, blur_sigma={blur_sigma}')

        self.align_size = align_size
        self.blur_sigma = blur_sigma
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

            # Get input configuration
            input_cfg = self.session.get_inputs()[0]
            input_shape = input_cfg.shape
            self.input_name = input_cfg.name

            # NHWC format: (N, H, W, C)
            if isinstance(input_shape[1], int) and isinstance(input_shape[2], int):
                self.input_size = (input_shape[2], input_shape[1])
            else:
                self.input_size = (256, 256)
                Logger.info(f'Dynamic input shape detected, using default: {self.input_size}')

            # Get output configuration
            outputs = self.session.get_outputs()
            self.output_names = [output.name for output in outputs]

            Logger.info(f'XSeg initialized with input size {self.input_size}')

        except Exception as e:
            Logger.error(f"Failed to load XSeg model from '{self.model_path}'", exc_info=True)
            raise RuntimeError(f'Failed to initialize XSeg model: {e}') from e

    def preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Preprocess an aligned face crop for inference.

        Args:
            face_crop (np.ndarray): An aligned face crop in BGR format.

        Returns:
            np.ndarray: Preprocessed image tensor with shape (1, H, W, 3).
        """
        # Resize to model input size
        image = cv2.resize(face_crop, self.input_size, interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Add batch dimension (NHWC format)
        image = np.expand_dims(image, axis=0)

        return image

    def postprocess(self, outputs: np.ndarray, crop_size: tuple[int, int]) -> np.ndarray:
        """
        Postprocess model output to segmentation mask.

        Args:
            outputs (np.ndarray): Raw model output.
            crop_size (tuple[int, int]): Size to resize mask to (width, height).

        Returns:
            np.ndarray: Segmentation mask as float32 in range [0, 1].
        """
        # Squeeze and clip to valid range
        mask = outputs.squeeze().clip(0, 1).astype(np.float32)

        # Resize back to crop size
        mask = cv2.resize(mask, crop_size, interpolation=cv2.INTER_LINEAR)

        # Apply optional blur and threshold
        if self.blur_sigma > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), self.blur_sigma)
            mask = (mask.clip(0.5, 1) - 0.5) * 2

        return mask

    def parse(self, image: np.ndarray, *, landmarks: np.ndarray | None = None) -> np.ndarray:
        """
        Perform face segmentation using 5-point landmarks.

        XSeg requires landmarks for face alignment. Unlike BiSeNet, calling
        this method without landmarks will raise a :class:`ValueError`.

        Args:
            image (np.ndarray): Input image in BGR format.
            landmarks (np.ndarray | None): 5-point facial landmarks with shape (5, 2).
                Required for XSeg face alignment.

        Returns:
            np.ndarray: Segmentation mask in original image space, values in [0, 1].

        Raises:
            ValueError: If landmarks is None or has incorrect shape.
        """
        if landmarks is None:
            raise ValueError(
                'XSeg requires 5-point facial landmarks for face alignment. Pass landmarks=... with shape (5, 2).'
            )
        if landmarks.shape != (5, 2):
            raise ValueError(f'Landmarks must have shape (5, 2), got {landmarks.shape}')

        # Align face using landmarks
        face_crop, inverse_matrix = face_alignment(image, landmarks, image_size=self.align_size)

        # Run inference
        crop_size = (face_crop.shape[1], face_crop.shape[0])
        input_tensor = self.preprocess(face_crop)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # Postprocess mask
        mask = self.postprocess(outputs[0], crop_size)

        # Warp mask back to original image space
        h, w = image.shape[:2]
        warped_mask = cv2.warpAffine(
            mask,
            inverse_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return warped_mask

    def parse_aligned(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Perform segmentation on an already aligned face crop.

        Args:
            face_crop (np.ndarray): An aligned face crop in BGR format.

        Returns:
            np.ndarray: Segmentation mask with same size as input, values in [0, 1].
        """
        crop_size = (face_crop.shape[1], face_crop.shape[0])

        # Run inference
        input_tensor = self.preprocess(face_crop)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        return self.postprocess(outputs[0], crop_size)

    def parse_with_inverse(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse face and return mask with inverse matrix for custom warping.

        Args:
            image (np.ndarray): Input image in BGR format.
            landmarks (np.ndarray): 5-point facial landmarks with shape (5, 2).

        Returns:
            Tuple of (mask, face_crop, inverse_matrix).
        """
        if landmarks.shape != (5, 2):
            raise ValueError(f'Landmarks must have shape (5, 2), got {landmarks.shape}')

        # Align face using landmarks
        face_crop, inverse_matrix = face_alignment(image, landmarks, image_size=self.align_size)

        # Run inference
        crop_size = (face_crop.shape[1], face_crop.shape[0])
        input_tensor = self.preprocess(face_crop)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # Postprocess mask (in crop space)
        mask = self.postprocess(outputs[0], crop_size)

        return mask, face_crop, inverse_matrix
