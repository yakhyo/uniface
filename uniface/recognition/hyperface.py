# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from uniface.constants import HyperFaceWeights
from uniface.model_store import verify_model_weights

from .base import BaseRecognizer, PreprocessConfig

__all__ = ['HyperFace']


class HyperFace(BaseRecognizer):
    """HyperFace: Generating Synthetic Face Recognition Datasets by Exploring Face Embedding Hypersphere.

    HyperFace is an IR-50 face encoder from Idiap, trained only on synthetic
    identities. It outputs 512-D embeddings from 112x112 aligned face crops.

    Key difference from other recognizers: HyperFace uses BGR color space
    (no RGB conversion) during preprocessing, like AdaFace.

    Args:
        model_name (HyperFaceWeights): The specific HyperFace model variant to use.
            Defaults to `HyperFaceWeights.IR_50`.
        preprocessing (PreprocessConfig | None): An optional custom preprocessing
            configuration. If None, a default config for HyperFace is used.
        providers (list[str] | None): ONNX Runtime execution providers. If None, auto-detects
            the best available provider. Example: ['CPUExecutionProvider'] to force CPU.

    Raises:
        ValueError: If the model weights are invalid or not found.
        RuntimeError: If the ONNX model fails to load or initialize.

    Example:
        >>> from uniface.recognition import HyperFace
        >>> recognizer = HyperFace()
        >>> # embedding = recognizer.get_normalized_embedding(image, landmarks)

    Reference:
        https://www.idiap.ch/paper/hyperface/
        https://github.com/yakhyo/hyperface-onnx
    """

    swap_rb = False

    def __init__(
        self,
        *,
        model_name: HyperFaceWeights = HyperFaceWeights.IR_50,
        preprocessing: PreprocessConfig | None = None,
        providers: list[str] | None = None,
    ) -> None:
        if preprocessing is None:
            preprocessing = PreprocessConfig(input_mean=127.5, input_std=127.5, input_size=(112, 112))
        model_path = verify_model_weights(model_name)
        super().__init__(model_path=model_path, preprocessing=preprocessing, providers=providers)
