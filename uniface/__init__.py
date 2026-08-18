# Copyright 2025-2026 Yakhyokhuja Valikhujaev
#
# Licensed under the MIT License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT

"""UniFace: A comprehensive library for face analysis.

This library provides unified APIs for:
- Face detection (BlazeFace, CenterFace, RetinaFace, SCRFD, YOLOv5Face, YOLOv8Face)
- Face recognition (AdaFace, ArcFace, EdgeFace, MobileFace, SphereFace)
- Face tracking (ByteTrack with Kalman filtering)
- Facial landmarks (106 / 98 / 68-point: 2d106det, PIPNet; 468/478-point dense: FaceMesh)
- Face parsing (semantic segmentation)
- Portrait matting (trimap-free alpha matte)
- Gaze estimation
- Head pose estimation
- Age, gender, emotion, and face state prediction (eyes open, glasses, mask)
- Face anti-spoofing
- Face image quality assessment (eDifFIQA)
- Privacy/anonymization
"""

from __future__ import annotations

__license__ = 'MIT'
__author__ = 'Yakhyokhuja Valikhujaev'
__version__ = '4.0.0'

from uniface.face_utils import compute_similarity, face_alignment
from uniface.log import Logger, enable_logging
from uniface.model_store import download_models, get_cache_dir, set_cache_dir, verify_model_weights

from .analyzer import FaceAnalyzer
from .attribute import AgeGender, Emotion, FaceAttribNet, FairFace
from .detection import SCRFD, BlazeFace, CenterFace, RetinaFace, YOLOv5Face, YOLOv8Face
from .gaze import MobileGaze
from .headpose import HeadPose
from .landmark import FaceMesh, Landmark106, PIPNet
from .matting import MODNet
from .parsing import BiSeNet, XSeg
from .privacy import BlurFace
from .quality import EDifFIQA
from .recognition import AdaFace, ArcFace, EdgeFace, MobileFace, SphereFace
from .spoofing import MiniFASNet

# The faiss dependency is imported lazily at FAISS(...) construction, so this
# import succeeds even without faiss-cpu installed.
from .stores import FAISS
from .tracking import BYTETracker
from .types import (
    DemographyResult,
    EmotionResult,
    Face,
    FaceMeshResult,
    FaceStateResult,
    GazeResult,
    HeadPoseResult,
    QualityResult,
    SpoofingResult,
)

__all__ = [
    # Metadata
    '__author__',
    '__license__',
    '__version__',
    # Core classes
    'Face',
    'FaceAnalyzer',
    # Detection models
    'BlazeFace',
    'CenterFace',
    'RetinaFace',
    'SCRFD',
    'YOLOv5Face',
    'YOLOv8Face',
    # Recognition models
    'AdaFace',
    'ArcFace',
    'EdgeFace',
    'MobileFace',
    'SphereFace',
    # Landmark models
    'FaceMesh',
    'FaceMeshResult',
    'Landmark106',
    'PIPNet',
    # Gaze models
    'GazeResult',
    'MobileGaze',
    # Head pose models
    'HeadPose',
    'HeadPoseResult',
    # Matting models
    'MODNet',
    # Parsing models
    'BiSeNet',
    'XSeg',
    # Attribute models
    'AgeGender',
    'DemographyResult',
    'Emotion',
    'EmotionResult',
    'FaceAttribNet',
    'FaceStateResult',
    'FairFace',
    # Spoofing models
    'MiniFASNet',
    'SpoofingResult',
    # Quality models
    'EDifFIQA',
    'QualityResult',
    # Tracking
    'BYTETracker',
    # Privacy
    'BlurFace',
    # Stores (optional)
    'FAISS',
    # Utilities
    'Logger',
    'compute_similarity',
    'download_models',
    'enable_logging',
    'face_alignment',
    'get_cache_dir',
    'set_cache_dir',
    'verify_model_weights',
]
