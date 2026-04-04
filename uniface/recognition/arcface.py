# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from uniface.constants import ArcFaceWeights
from uniface.model_store import verify_model_weights

from .base import BaseRecognizer, PreprocessConfig

__all__ = ['ArcFace']


class ArcFace(BaseRecognizer):
    """ArcFace model for robust face recognition.

    This class provides a concrete implementation of the BaseRecognizer,
    pre-configured for ArcFace models. It handles the loading of specific
    ArcFace weights and sets up the appropriate default preprocessing.

    Args:
        model_name (ArcFaceWeights): The specific ArcFace model variant to use.
            Defaults to `ArcFaceWeights.MNET`.
        preprocessing (Optional[PreprocessConfig]): An optional custom preprocessing
            configuration. If None, a default config for ArcFace is used.
        providers (list[str] | None): ONNX Runtime execution providers. If None, auto-detects
            the best available provider. Example: ['CPUExecutionProvider'] to force CPU.

    Example:
        >>> from uniface.recognition import ArcFace
        >>> recognizer = ArcFace()
        >>> # embedding = recognizer.get_normalized_embedding(image, landmarks)

    Reference:
        https://arxiv.org/abs/1801.07698
        https://github.com/yakhyo/face-reidentification
    """

    def __init__(
        self,
        model_name: ArcFaceWeights = ArcFaceWeights.MNET,
        preprocessing: PreprocessConfig | None = None,
        providers: list[str] | None = None,
    ) -> None:
        if preprocessing is None:
            preprocessing = PreprocessConfig(input_mean=127.5, input_std=127.5, input_size=(112, 112))
        model_path = verify_model_weights(model_name)
        super().__init__(model_path=model_path, preprocessing=preprocessing, providers=providers)
