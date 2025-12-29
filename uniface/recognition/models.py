# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from uniface.constants import ArcFaceWeights, MobileFaceWeights, SphereFaceWeights
from uniface.model_store import verify_model_weights

from .base import BaseRecognizer, PreprocessConfig

__all__ = ['ArcFace', 'MobileFace', 'SphereFace']


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

    Example:
        >>> from uniface.recognition import ArcFace
        >>> recognizer = ArcFace()
        >>> # embedding = recognizer.get_normalized_embedding(image, landmarks)
    """

    def __init__(
        self,
        model_name: ArcFaceWeights = ArcFaceWeights.MNET,
        preprocessing: PreprocessConfig | None = None,
    ) -> None:
        if preprocessing is None:
            preprocessing = PreprocessConfig(input_mean=127.5, input_std=127.5, input_size=(112, 112))
        model_path = verify_model_weights(model_name)
        super().__init__(model_path=model_path, preprocessing=preprocessing)


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

    Example:
        >>> from uniface.recognition import MobileFace
        >>> recognizer = MobileFace()
        >>> # embedding = recognizer.get_normalized_embedding(image, landmarks)
    """

    def __init__(
        self,
        model_name: MobileFaceWeights = MobileFaceWeights.MNET_V2,
        preprocessing: PreprocessConfig | None = None,
    ) -> None:
        if preprocessing is None:
            preprocessing = PreprocessConfig(input_mean=127.5, input_std=127.5, input_size=(112, 112))
        model_path = verify_model_weights(model_name)
        super().__init__(model_path=model_path, preprocessing=preprocessing)


class SphereFace(BaseRecognizer):
    """SphereFace model using angular margin for face recognition.

    This class provides a concrete implementation of the BaseRecognizer,
    pre-configured for SphereFace models, which were among the first to
    introduce angular margin loss functions.

    Args:
        model_name (SphereFaceWeights): The specific SphereFace model variant to use.
            Defaults to `SphereFaceWeights.SPHERE20`.
        preprocessing (Optional[PreprocessConfig]): An optional custom preprocessing
            configuration. If None, a default config for SphereFace is used.

    Example:
        >>> from uniface.recognition import SphereFace
        >>> recognizer = SphereFace()
        >>> # embedding = recognizer.get_normalized_embedding(image, landmarks)
    """

    def __init__(
        self,
        model_name: SphereFaceWeights = SphereFaceWeights.SPHERE20,
        preprocessing: PreprocessConfig | None = None,
    ) -> None:
        if preprocessing is None:
            preprocessing = PreprocessConfig(input_mean=127.5, input_std=127.5, input_size=(112, 112))

        model_path = verify_model_weights(model_name)
        super().__init__(model_path=model_path, preprocessing=preprocessing)
