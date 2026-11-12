# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

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
            - IR_50: Mid-size model trained on WebFace4M (166 MB)
            - IR_101: Larger model trained on WebFace12M (249 MB)
            Defaults to `AdaFaceWeights.IR_18`.
        preprocessing (PreprocessConfig | None): An optional custom preprocessing
            configuration. If None, a default config for AdaFace is used.
        providers (list[str] | None): ONNX Runtime execution providers. If None, auto-detects
            the best available provider. Example: ['CPUExecutionProvider'] to force CPU.

    Raises:
        ValueError: If the model weights are invalid or not found.
        RuntimeError: If the ONNX model fails to load or initialize.

    Example:
        >>> from uniface.recognition import AdaFace
        >>> recognizer = AdaFace()
        >>> # embedding = recognizer.get_normalized_embedding(image, landmarks)

    Reference:
        https://github.com/mk-minchul/AdaFace
        https://github.com/yakhyo/adaface-onnx
    """

    swap_rb = False

    def __init__(
        self,
        *,
        model_name: AdaFaceWeights = AdaFaceWeights.IR_18,
        preprocessing: PreprocessConfig | None = None,
        providers: list[str] | None = None,
    ) -> None:
        if preprocessing is None:
            preprocessing = PreprocessConfig(input_mean=127.5, input_std=127.5, input_size=(112, 112))
        model_path = verify_model_weights(model_name)
        super().__init__(model_path=model_path, preprocessing=preprocessing, providers=providers)
