# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import cv2
import numpy as np

from uniface.constants import AdaFaceWeights
from uniface.model_store import verify_model_weights

from .base import BaseRecognizer, PreprocessConfig

__all__ = ['AdaFace']


class AdaFace(BaseRecognizer):
    """AdaFace model for high-quality face recognition.

    AdaFace introduces adaptive margin based on image quality, achieving
    state-of-the-art results on challenging benchmarks like IJB-B and IJB-C.

    Key difference from other recognizers: AdaFace uses BGR color space
    (no RGB conversion) during preprocessing.

    Args:
        model_name (AdaFaceWeights): The specific AdaFace model variant to use.
            - IR_18: Smaller model trained on WebFace4M (92 MB)
            - IR_101: Larger model trained on WebFace12M (249 MB)
            Defaults to `AdaFaceWeights.IR_18`.
        preprocessing (Optional[PreprocessConfig]): An optional custom preprocessing
            configuration. If None, a default config for AdaFace is used.
        providers (list[str] | None): ONNX Runtime execution providers. If None, auto-detects
            the best available provider. Example: ['CPUExecutionProvider'] to force CPU.

    Example:
        >>> from uniface.recognition import AdaFace
        >>> recognizer = AdaFace()
        >>> # embedding = recognizer.get_normalized_embedding(image, landmarks)

    Reference:
        https://github.com/mk-minchul/AdaFace
        https://github.com/yakhyo/adaface-onnx
    """

    def __init__(
        self,
        model_name: AdaFaceWeights = AdaFaceWeights.IR_18,
        preprocessing: PreprocessConfig | None = None,
        providers: list[str] | None = None,
    ) -> None:
        if preprocessing is None:
            preprocessing = PreprocessConfig(input_mean=127.5, input_std=127.5, input_size=(112, 112))
        model_path = verify_model_weights(model_name)
        super().__init__(model_path=model_path, preprocessing=preprocessing, providers=providers)

    def preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocess the image: resize, normalize, and convert to blob.

        AdaFace uses BGR color space (no RGB conversion).

        Args:
            face_img: Input image in BGR format.

        Returns:
            Preprocessed image as a NumPy array ready for inference.
        """
        resized_img = cv2.resize(face_img, self.input_size)

        if isinstance(self.input_std, list | tuple):
            # Per-channel normalization (keep BGR)
            mean_array = np.array(self.input_mean, dtype=np.float32)
            std_array = np.array(self.input_std, dtype=np.float32)
            normalized_img = (resized_img.astype(np.float32) - mean_array) / std_array

            # Change to NCHW format (batch, channels, height, width)
            blob = np.transpose(normalized_img, (2, 0, 1))  # CHW
            blob = np.expand_dims(blob, axis=0)  # NCHW
        else:
            # Single-value normalization using cv2.dnn (keep BGR, swapRB=False)
            blob = cv2.dnn.blobFromImage(
                resized_img,
                scalefactor=1.0 / self.input_std,
                size=self.input_size,
                mean=(self.input_mean, self.input_mean, self.input_mean),
                swapRB=False,  # Keep BGR for AdaFace
            )

        return blob
