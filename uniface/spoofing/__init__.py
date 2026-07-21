# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from uniface.constants import MiniFASNetWeights
from uniface.types import SpoofingResult

from .base import BaseSpoofer
from .minifasnet import MiniFASNet

__all__ = [
    'BaseSpoofer',
    'MiniFASNet',
    'MiniFASNetWeights',
    'SpoofingResult',
]
