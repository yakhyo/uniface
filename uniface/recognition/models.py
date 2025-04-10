# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from typing import Optional

from uniface.constants import SphereFaceWeights, MobileFaceWeights, ArcFaceWeights
from .base import BaseFaceEncoder, PreprocessConfig


__all__ = ["SphereFace", "MobileFace", "ArcFace"]


class SphereFace(BaseFaceEncoder):
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


class MobileFace(BaseFaceEncoder):
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


class ArcFace(BaseFaceEncoder):
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
