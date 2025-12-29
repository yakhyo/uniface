# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from abc import ABC, abstractmethod

import numpy as np


class BaseFaceParser(ABC):
    """
    Abstract base class for all face parsing models.

    This class defines the common interface that all face parsing models must implement,
    ensuring consistency across different parsing methods. Face parsing segments a face
    image into semantic regions such as skin, eyes, nose, mouth, hair, etc.

    The output is a segmentation mask where each pixel is assigned a class label
    representing a facial component.
    """

    @abstractmethod
    def _initialize_model(self) -> None:
        """
        Initialize the underlying model for inference.

        This method should handle loading model weights, creating the
        inference session (e.g., ONNX Runtime), and any necessary
        setup procedures to prepare the model for prediction.

        Raises:
            RuntimeError: If the model fails to load or initialize.
        """
        raise NotImplementedError('Subclasses must implement the _initialize_model method.')

    @abstractmethod
    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input face image for model inference.

        This method should take a raw face crop and convert it into the format
        expected by the model's inference engine (e.g., normalized tensor).

        Args:
            face_image (np.ndarray): A face image in BGR format with
                                     shape (H, W, C).

        Returns:
            np.ndarray: The preprocessed image tensor ready for inference,
                        typically with shape (1, C, H, W).
        """
        raise NotImplementedError('Subclasses must implement the preprocess method.')

    @abstractmethod
    def postprocess(self, outputs: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
        """
        Postprocess raw model outputs into a segmentation mask.

        This method takes the raw output from the model's inference and
        converts it into a segmentation mask at the original image size.

        Args:
            outputs (np.ndarray): Raw outputs from the model inference.
            original_size (Tuple[int, int]): Original image size (width, height).

        Returns:
            np.ndarray: Segmentation mask with the same size as the original image.
        """
        raise NotImplementedError('Subclasses must implement the postprocess method.')

    @abstractmethod
    def parse(self, face_image: np.ndarray) -> np.ndarray:
        """
        Perform end-to-end face parsing on a face image.

        This method orchestrates the full pipeline: preprocessing the input,
        running inference, and postprocessing to return the segmentation mask.

        Args:
            face_image (np.ndarray): A face image in BGR format.
                                     The face should be roughly centered and
                                     well-framed within the image.

        Returns:
            np.ndarray: Segmentation mask with the same size as input image,
                       where each pixel value represents a facial component class.

        Example:
            >>> parser = create_face_parser()
            >>> mask = parser.parse(face_crop)
            >>> print(f'Mask shape: {mask.shape}, unique classes: {np.unique(mask)}')
        """
        raise NotImplementedError('Subclasses must implement the parse method.')

    def __call__(self, face_image: np.ndarray) -> np.ndarray:
        """
        Provides a convenient, callable shortcut for the `parse` method.

        Args:
            face_image (np.ndarray): A face image in BGR format.

        Returns:
            np.ndarray: Segmentation mask with the same size as input image.
        """
        return self.parse(face_image)
