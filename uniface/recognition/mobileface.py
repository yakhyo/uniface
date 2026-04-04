# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from uniface.constants import MobileFaceWeights
from uniface.model_store import verify_model_weights

from .base import BaseRecognizer, PreprocessConfig

__all__ = ['MobileFace']


class MobileFace(BaseRecognizer):
    """Lightweight MobileFaceNet model for fast face recognition.

    This class provides a concrete implementation of the BaseRecognizer,
    pre-configured for MobileFaceNet models. It is optimized for speed,
    making it suitable for edge devices.

    Args:
        model_name (MobileFaceWeights): The specific MobileFaceNet model variant to use.
            Defaults to `MobileFaceWeights.MNET_V2`.
        preprocessing (Optional[PreprocessConfig]): An optional custom preprocessing
            configuration. If None, a default config for MobileFaceNet is used.
        providers (list[str] | None): ONNX Runtime execution providers. If None, auto-detects
            the best available provider. Example: ['CPUExecutionProvider'] to force CPU.

    Example:
        >>> from uniface.recognition import MobileFace
        >>> recognizer = MobileFace()
        >>> # embedding = recognizer.get_normalized_embedding(image, landmarks)

    Reference:
        https://arxiv.org/abs/1804.07573
        https://github.com/yakhyo/face-recognition
    """

    def __init__(
        self,
        model_name: MobileFaceWeights = MobileFaceWeights.MNET_V2,
        preprocessing: PreprocessConfig | None = None,
        providers: list[str] | None = None,
    ) -> None:
        if preprocessing is None:
            preprocessing = PreprocessConfig(input_mean=127.5, input_std=127.5, input_size=(112, 112))
        model_path = verify_model_weights(model_name)
        super().__init__(model_path=model_path, preprocessing=preprocessing, providers=providers)
