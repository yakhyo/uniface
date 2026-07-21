# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from .base import BaseFaceParser
from .bisenet import BiSeNet
from .xseg import XSeg

__all__ = ['BaseFaceParser', 'BiSeNet', 'XSeg']
