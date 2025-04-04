# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from typing import Optional

from .base import BaseFaceEncoder, PreprocessConfig
from uniface.constants import SphereFaceWeights, MobileFaceWeights, ArcFaceWeights


__all__ = ["SphereFace", "MobileFace", "ArcFace"]


class SphereFace(BaseFaceEncoder):
    def __init__(
        self, model_path: Optional[SphereFaceWeights] = SphereFaceWeights.SPHERE20,
        preprocessing: Optional[PreprocessConfig] = None
    ) -> None:
        if preprocessing is None:
            preprocessing = PreprocessConfig(
                input_mean=127.5,
                input_std=127.5,
                input_size=(112, 112)
            )
        super().__init__(model_path=model_path, preprocessing=preprocessing)


class MobileFace(BaseFaceEncoder):
    def __init__(
        self, model_path: Optional[MobileFaceWeights] = MobileFaceWeights.MNET_V2,
        preprocessing: Optional[PreprocessConfig] = None
    ) -> None:
        if preprocessing is None:
            preprocessing = PreprocessConfig(
                input_mean=127.5,
                input_std=127.5,
                input_size=(112, 112)
            )
        super().__init__(model_path=model_path)


class ArcFace(BaseFaceEncoder):
    def __init__(
        self, model_path: Optional[ArcFaceWeights] = ArcFaceWeights.MNET,
        preprocessing: Optional[PreprocessConfig] = None
    ) -> None:
        if preprocessing is None:
            preprocessing = PreprocessConfig(
                input_mean=127.5,
                input_std=127.5,
                input_size=(112, 112)
            )
        super().__init__(model_path=model_path)
