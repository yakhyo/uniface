# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from typing import Optional

from .base import BaseFaceEncoder
from uniface.constants import SphereFaceWeights, MobileFaceWeights, ArcFaceWeights


__all__ = ["SphereFace", "MobileFace", "ArcFace"]


class SphereFace(BaseFaceEncoder):
    def __init__(self, model_path: Optional[SphereFaceWeights] = SphereFaceWeights.SPHERE20) -> None:
        super().__init__(model_path=model_path)


class MobileFace(BaseFaceEncoder):
    def __init__(self, model_path: Optional[MobileFaceWeights] = MobileFaceWeights.MNET_V2) -> None:
        super().__init__(model_path=model_path)


class ArcFace(BaseFaceEncoder):
    def __init__(self, model_path: Optional[ArcFaceWeights] = ArcFaceWeights.MNET) -> None:
        super().__init__(model_path=model_path)
