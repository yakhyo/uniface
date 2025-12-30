# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


import cv2
import numpy as np

from uniface.constants import MiniFASNetWeights
from uniface.log import Logger
from uniface.model_store import verify_model_weights
from uniface.onnx_utils import create_onnx_session
from uniface.types import SpoofingResult

from .base import BaseSpoofer

__all__ = ['MiniFASNet']

# Default crop scales for each model variant
DEFAULT_SCALES = {
    MiniFASNetWeights.V1SE: 4.0,
    MiniFASNetWeights.V2: 2.7,
}


class MiniFASNet(BaseSpoofer):
    """
    MiniFASNet: Lightweight Face Anti-Spoofing with ONNX Runtime.

    MiniFASNet is a face anti-spoofing model that detects whether a face is real
    (live person) or fake (photo, video replay, mask, etc.). It supports two model
    variants: V1SE (with squeeze-and-excitation) and V2 (improved version).

    The model takes a face region cropped from the image using a bounding box
    and predicts whether it's a real or spoofed face.

    Reference:
        https://github.com/yakhyo/face-anti-spoofing

    Args:
        model_name (MiniFASNetWeights): The enum specifying the model variant to load.
            Options: V1SE (scale=4.0), V2 (scale=2.7).
            Defaults to `MiniFASNetWeights.V2`.
        scale (Optional[float]): Custom crop scale factor for face region.
            If None, uses the default scale for the selected model variant.
            V1SE uses 4.0, V2 uses 2.7.

    Attributes:
        scale (float): Crop scale factor for face region extraction.
        input_size (Tuple[int, int]): Model input dimensions (width, height).

    Example:
        >>> from uniface.spoofing import MiniFASNet
        >>> from uniface import RetinaFace
        >>>
        >>> detector = RetinaFace()
        >>> spoofer = MiniFASNet()
        >>>
        >>> # Detect faces and check if they are real
        >>> faces = detector.detect(image)
        >>> for face in faces:
        ...     result = spoofer.predict(image, face.bbox)
        ...     label = 'Real' if result.is_real else 'Fake'
        ...     print(f'{label}: {result.confidence:.2%}')
    """

    def __init__(
        self,
        model_name: MiniFASNetWeights = MiniFASNetWeights.V2,
        scale: float | None = None,
    ) -> None:
        Logger.info(f'Initializing MiniFASNet with model={model_name.name}')

        # Use default scale for the model variant if not specified
        self.scale = scale if scale is not None else DEFAULT_SCALES.get(model_name, 2.7)

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
            self.input_name = input_cfg.name
            # Input shape is (batch, channels, height, width) - we need (width, height)
            self.input_size = tuple(input_cfg.shape[2:4][::-1])  # (width, height)

            # Get output configuration
            output_cfg = self.session.get_outputs()[0]
            self.output_name = output_cfg.name

            Logger.info(f'MiniFASNet initialized with input size {self.input_size}, scale={self.scale}')

        except Exception as e:
            Logger.error(f"Failed to load MiniFASNet model from '{self.model_path}'", exc_info=True)
            raise RuntimeError(f'Failed to initialize MiniFASNet model: {e}') from e

    def _xyxy_to_xywh(self, bbox: list | np.ndarray) -> list[int]:
        """Convert bounding box from [x1, y1, x2, y2] to [x, y, w, h] format."""
        x1, y1, x2, y2 = bbox[:4]
        return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

    def _crop_face(self, image: np.ndarray, bbox_xywh: list[int]) -> np.ndarray:
        """
        Crop and resize face region from image using scale factor.

        The crop is centered on the face bounding box and scaled to capture
        more context around the face, which is important for anti-spoofing.

        Args:
            image: Input image in BGR format.
            bbox_xywh: Face bounding box in [x, y, w, h] format.

        Returns:
            Cropped and resized face region.
        """
        src_h, src_w = image.shape[:2]
        x, y, box_w, box_h = bbox_xywh

        # Calculate the scale to apply based on image and face size
        scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, self.scale)
        new_w = box_w * scale
        new_h = box_h * scale

        # Calculate center of the bounding box
        center_x = x + box_w / 2
        center_y = y + box_h / 2

        # Calculate new bounding box coordinates
        x1 = max(0, int(center_x - new_w / 2))
        y1 = max(0, int(center_y - new_h / 2))
        x2 = min(src_w - 1, int(center_x + new_w / 2))
        y2 = min(src_h - 1, int(center_y + new_h / 2))

        # Crop and resize
        cropped = image[y1 : y2 + 1, x1 : x2 + 1]
        resized = cv2.resize(cropped, self.input_size)

        return resized

    def preprocess(self, image: np.ndarray, bbox: list | np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for model inference.

        Crops the face region, converts to float32, and arranges
        dimensions for the model (NCHW format).

        Args:
            image: Input image in BGR format with shape (H, W, C).
            bbox: Face bounding box in [x1, y1, x2, y2] format.

        Returns:
            Preprocessed image tensor with shape (1, C, H, W).
        """
        # Convert bbox format
        bbox_xywh = self._xyxy_to_xywh(bbox)

        # Crop and resize face region
        face = self._crop_face(image, bbox_xywh)

        # Convert to float32 (no normalization needed for this model)
        face = face.astype(np.float32)

        # HWC -> CHW -> NCHW
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)

        return face

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to logits along axis 1."""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def postprocess(self, outputs: np.ndarray) -> SpoofingResult:
        """
        Postprocess raw model outputs into prediction result.

        Applies softmax to convert logits to probabilities and
        returns the SpoofingResult with is_real flag and confidence score.

        Args:
            outputs: Raw outputs from the model inference (logits).

        Returns:
            SpoofingResult: Result containing is_real flag and confidence score.
        """
        probs = self._softmax(outputs)
        label_idx = int(np.argmax(probs))
        confidence = float(probs[0, label_idx])

        return SpoofingResult(is_real=(label_idx == 1), confidence=confidence)

    def predict(self, image: np.ndarray, bbox: list | np.ndarray) -> SpoofingResult:
        """
        Perform end-to-end anti-spoofing prediction on a face.

        Args:
            image: Input image in BGR format containing the face.
            bbox: Face bounding box in [x1, y1, x2, y2] format.

        Returns:
            SpoofingResult: Result containing is_real flag and confidence score.
        """
        # Preprocess
        input_tensor = self.preprocess(image, bbox)

        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0]

        # Postprocess and return
        return self.postprocess(outputs)
