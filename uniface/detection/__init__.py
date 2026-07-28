# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from .base import BaseDetector
from .blazeface import BlazeFace
from .centerface import CenterFace
from .retinaface import RetinaFace
from .scrfd import SCRFD
from .yolov5 import YOLOv5Face
from .yolov8 import YOLOv8Face

__all__ = [
    'SCRFD',
    'BaseDetector',
    'BlazeFace',
    'CenterFace',
    'RetinaFace',
    'YOLOv5Face',
    'YOLOv8Face',
]
