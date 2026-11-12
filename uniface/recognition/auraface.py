# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from uniface.constants import AuraFaceWeights
from uniface.model_store import verify_model_weights

from .base import BaseRecognizer, PreprocessConfig

__all__ = ['AuraFace']


class AuraFace(BaseRecognizer):
    """AuraFace: Open-Source Face Recognition and Identity Preservation Models.

    AuraFace is a ResNet100 face encoder released by fal. It outputs 512-D
    embeddings from 112x112 aligned face crops.

    Args:
        model_name (AuraFaceWeights): The specific AuraFace model variant to use.
            Defaults to `AuraFaceWeights.R100`.
        preprocessing (PreprocessConfig | None): An optional custom preprocessing
            configuration. If None, a default config for AuraFace is used.
        providers (list[str] | None): ONNX Runtime execution providers. If None, auto-detects
            the best available provider. Example: ['CPUExecutionProvider'] to force CPU.

    Raises:
        ValueError: If the model weights are invalid or not found.
        RuntimeError: If the ONNX model fails to load or initialize.

    Example:
        >>> from uniface.recognition import AuraFace
        >>> recognizer = AuraFace()
        >>> # embedding = recognizer.get_normalized_embedding(image, landmarks)

    Reference:
        https://huggingface.co/fal/AuraFace-v1
        https://github.com/yakhyo/auraface-onnx
    """

    def __init__(
        self,
        *,
        model_name: AuraFaceWeights = AuraFaceWeights.R100,
        preprocessing: PreprocessConfig | None = None,
        providers: list[str] | None = None,
    ) -> None:
        if preprocessing is None:
            preprocessing = PreprocessConfig(input_mean=127.5, input_std=127.5, input_size=(112, 112))
        model_path = verify_model_weights(model_name)
        super().__init__(model_path=model_path, preprocessing=preprocessing, providers=providers)
