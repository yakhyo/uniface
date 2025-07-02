# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from typing import Optional

from uniface.constants import SphereFaceWeights, MobileFaceWeights, ArcFaceWeights
from .base import BaseModel, PreprocessConfig


__all__ = ["SphereFace", "MobileFace", "ArcFace"]


class SphereFace(BaseModel):
    """
    SphereFace face encoder class.

    This class loads a SphereFace model for face embedding extraction.
    It supports configurable preprocessing, with a default mean/std and input size of 112x112.

    Args:
        model_name (SphereFaceWeights): Enum value representing the model to load. Defaults to SphereFaceWeights.SPHERE20.
        preprocessing (Optional[PreprocessConfig]): Preprocessing config (mean, std, size). Defaults to standard 112x112 with normalization.
    """

    def __init__(
        self, model_name: SphereFaceWeights = SphereFaceWeights.SPHERE20,
        preprocessing: Optional[PreprocessConfig] = None
    ) -> None:
        if preprocessing is None:
            preprocessing = PreprocessConfig(
                input_mean=127.5,
                input_std=127.5,
                input_size=(112, 112)
            )
        super().__init__(model_name=model_name, preprocessing=preprocessing)


class MobileFace(BaseModel):
    """
    MobileFace face encoder class.

    Loads a lightweight MobileFaceNet model for fast face embedding extraction.
    Default input normalization and resizing applied if preprocessing is not provided.

    Args:
        model_name (MobileFaceWeights): Enum value specifying the MobileFace model. Defaults to MobileFaceWeights.MNET_V2.
        preprocessing (Optional[PreprocessConfig]): Preprocessing config. If None, uses standard normalization and 112x112 input size.
    """

    def __init__(
        self, model_name: MobileFaceWeights = MobileFaceWeights.MNET_V2,
        preprocessing: Optional[PreprocessConfig] = None
    ) -> None:
        if preprocessing is None:
            preprocessing = PreprocessConfig(
                input_mean=127.5,
                input_std=127.5,
                input_size=(112, 112)
            )
        super().__init__(model_name=model_name)


class ArcFace(BaseModel):
    """
    ArcFace face encoder class.

    Loads an ArcFace model (e.g., ResNet-based) for robust face recognition embedding generation.
    Applies standard preprocessing unless overridden.

    Args:
        model_name (ArcFaceWeights): Enum for the ArcFace model variant. Defaults to ArcFaceWeights.MNET.
        preprocessing (Optional[PreprocessConfig]): Preprocessing settings. Defaults to standard normalization and resizing if not specified.
    """

    def __init__(
        self, model_name: ArcFaceWeights = ArcFaceWeights.MNET,
        preprocessing: Optional[PreprocessConfig] = None
    ) -> None:
        if preprocessing is None:
            preprocessing = PreprocessConfig(
                input_mean=127.5,
                input_std=127.5,
                input_size=(112, 112)
            )
        super().__init__(model_name=model_name)
