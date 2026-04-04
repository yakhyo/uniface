# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from uniface.constants import EdgeFaceWeights
from uniface.model_store import verify_model_weights

from .base import BaseRecognizer, PreprocessConfig

__all__ = ['EdgeFace']


class EdgeFace(BaseRecognizer):
    """EdgeFace: Efficient Face Recognition Model for Edge Devices.

    EdgeFace uses an EdgeNeXt backbone with optional LoRA low-rank
    compression, offering a strong accuracy-efficiency trade-off for
    deployment on resource-constrained hardware. Competition-winning
    entry (compact track) at EFaR 2023, IJCB.

    All variants output 512-D embeddings from 112x112 aligned face crops.

    Args:
        model_name (EdgeFaceWeights): The specific EdgeFace model variant to use.
            - XXS: Ultra-compact (1.24M params, ~5 MB)
            - XS_GAMMA_06: Compact with LoRA (1.77M params, ~7 MB)
            - S_GAMMA_05: Small with LoRA (3.65M params, ~14 MB)
            - BASE: Full-size model (18.2M params, ~70 MB)
            Defaults to `EdgeFaceWeights.XXS`.
        preprocessing (Optional[PreprocessConfig]): An optional custom preprocessing
            configuration. If None, a default config for EdgeFace is used.
        providers (list[str] | None): ONNX Runtime execution providers. If None, auto-detects
            the best available provider. Example: ['CPUExecutionProvider'] to force CPU.

    Example:
        >>> from uniface.recognition import EdgeFace
        >>> recognizer = EdgeFace()
        >>> # embedding = recognizer.get_normalized_embedding(image, landmarks)

    Reference:
        https://arxiv.org/abs/2307.01838v2
        https://github.com/otroshi/edgeface
        https://github.com/yakhyo/edgeface-onnx
    """

    def __init__(
        self,
        model_name: EdgeFaceWeights = EdgeFaceWeights.XXS,
        preprocessing: PreprocessConfig | None = None,
        providers: list[str] | None = None,
    ) -> None:
        if preprocessing is None:
            preprocessing = PreprocessConfig(input_mean=127.5, input_std=127.5, input_size=(112, 112))
        model_path = verify_model_weights(model_name)
        super().__init__(model_path=model_path, preprocessing=preprocessing, providers=providers)
