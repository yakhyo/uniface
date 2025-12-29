# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


import cv2
import numpy as np

from uniface.constants import ParsingWeights
from uniface.log import Logger
from uniface.model_store import verify_model_weights
from uniface.onnx_utils import create_onnx_session

from .base import BaseFaceParser

__all__ = ['BiSeNet']


class BiSeNet(BaseFaceParser):
    """
    BiSeNet: Bilateral Segmentation Network for Face Parsing with ONNX Runtime.

    BiSeNet is a semantic segmentation model that segments a face image into
    different facial components such as skin, eyes, nose, mouth, hair, etc. The model
    uses a BiSeNet architecture with ResNet backbone and outputs a segmentation mask
    where each pixel is assigned a class label.

    The model supports 19 facial component classes including:
    - Background, skin, eyebrows, eyes, nose, mouth, lips, ears, hair, etc.

    Reference:
        https://github.com/yakhyo/face-parsing

    Args:
        model_name (ParsingWeights): The enum specifying the parsing model to load.
            Options: RESNET18, RESNET34.
            Defaults to `ParsingWeights.RESNET18`.
        input_size (Tuple[int, int]): The resolution (width, height) for the model's
            input. Defaults to (512, 512).

    Attributes:
        input_size (Tuple[int, int]): Model input dimensions.
        input_mean (np.ndarray): Per-channel mean values for normalization (ImageNet).
        input_std (np.ndarray): Per-channel std values for normalization (ImageNet).

    Example:
        >>> from uniface.parsing import BiSeNet
        >>> from uniface import RetinaFace
        >>>
        >>> detector = RetinaFace()
        >>> parser = BiSeNet()
        >>>
        >>> # Detect faces and parse each face
        >>> faces = detector.detect(image)
        >>> for face in faces:
        ...     bbox = face.bbox
        ...     x1, y1, x2, y2 = map(int, bbox[:4])
        ...     face_crop = image[y1:y2, x1:x2]
        ...     mask = parser.parse(face_crop)
        ...     print(f'Mask shape: {mask.shape}, unique classes: {np.unique(mask)}')
    """

    def __init__(
        self,
        model_name: ParsingWeights = ParsingWeights.RESNET18,
        input_size: tuple[int, int] = (512, 512),
    ) -> None:
        Logger.info(f'Initializing BiSeNet with model={model_name}, input_size={input_size}')

        self.input_size = input_size
        self.input_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.input_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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

            Logger.info(f'BiSeNet initialized with input size {self.input_size}')

        except Exception as e:
            Logger.error(f"Failed to load parsing model from '{self.model_path}'", exc_info=True)
            raise RuntimeError(f'Failed to initialize parsing model: {e}') from e

    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess a face image for parsing.

        Args:
            face_image (np.ndarray): A face image in BGR format.

        Returns:
            np.ndarray: Preprocessed image tensor with shape (1, 3, H, W).
        """
        # Convert BGR to RGB
        image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1] and apply normalization
        image = image.astype(np.float32) / 255.0
        image = (image - self.input_mean) / self.input_std

        # HWC -> CHW -> NCHW
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0).astype(np.float32)

        return image

    def postprocess(self, outputs: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
        """
        Postprocess model output to segmentation mask.

        Args:
            outputs (np.ndarray): Raw model output.
            original_size (Tuple[int, int]): Original image size (width, height).

        Returns:
            np.ndarray: Segmentation mask resized to original dimensions.
        """
        # Get the class with highest probability for each pixel
        predicted_mask = outputs.squeeze(0).argmax(0).astype(np.uint8)

        # Resize back to original size
        restored_mask = cv2.resize(predicted_mask, original_size, interpolation=cv2.INTER_NEAREST)

        return restored_mask

    def parse(self, face_image: np.ndarray) -> np.ndarray:
        """
        Perform end-to-end face parsing on a face image.

        This method orchestrates the full pipeline: preprocessing the input,
        running inference, and postprocessing to return the segmentation mask.

        Args:
            face_image (np.ndarray): A face image in BGR format.

        Returns:
            np.ndarray: Segmentation mask with the same size as input image.
        """
        original_size = (face_image.shape[1], face_image.shape[0])  # (width, height)
        input_tensor = self.preprocess(face_image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        return self.postprocess(outputs[0], original_size)
