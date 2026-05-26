# Copyright 2025-2026 Yakhyokhuja Valikhujaev
#
# Licensed under the MIT License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""UniFace: A comprehensive library for face analysis.

This library provides unified APIs for:
- Face detection (RetinaFace, SCRFD, YOLOv5Face, YOLOv8Face)
- Face recognition (AdaFace, ArcFace, EdgeFace, MobileFace, SphereFace)
- Face tracking (ByteTrack with Kalman filtering)
- Facial landmarks (106 / 98 / 68-point detection: 2d106det, PIPNet)
- Face parsing (semantic segmentation)
- Portrait matting (trimap-free alpha matte)
- Gaze estimation
- Head pose estimation
- Age, gender, and emotion prediction
- Face anti-spoofing
- Face image quality assessment (eDifFIQA)
- Privacy/anonymization
"""

from __future__ import annotations

__license__ = 'MIT'
__author__ = 'Yakhyokhuja Valikhujaev'
__version__ = '3.7.0rc1'

import contextlib

from uniface.face_utils import compute_similarity, face_alignment
from uniface.log import Logger, enable_logging
from uniface.model_store import download_models, get_cache_dir, set_cache_dir, verify_model_weights

from .analyzer import FaceAnalyzer
from .attribute import AgeGender, Emotion, FairFace, create_attribute_predictor
from .detection import (
    SCRFD,
    RetinaFace,
    YOLOv5Face,
    YOLOv8Face,
    create_detector,
    list_available_detectors,
)
from .gaze import MobileGaze, create_gaze_estimator
from .headpose import HeadPose, create_head_pose_estimator
from .landmark import Landmark106, PIPNet, create_landmarker
from .matting import MODNet, create_matting_model
from .parsing import BiSeNet, XSeg, create_face_parser
from .privacy import BlurFace
from .quality import EDifFIQA
from .recognition import AdaFace, ArcFace, EdgeFace, MobileFace, SphereFace, create_recognizer
from .spoofing import MiniFASNet, create_spoofer
from .tracking import BYTETracker
from .types import AttributeResult, EmotionResult, Face, GazeResult, HeadPoseResult, QualityResult, SpoofingResult

# Optional: FAISS vector store (requires `pip install faiss-cpu`)
with contextlib.suppress(ImportError):
    from .stores import FAISS

__all__ = [
    # Metadata
    '__author__',
    '__license__',
    '__version__',
    # Core classes
    'Face',
    'FaceAnalyzer',
    # Factory functions
    'create_detector',
    'create_face_parser',
    'create_gaze_estimator',
    'create_matting_model',
    'create_head_pose_estimator',
    'create_landmarker',
    'create_recognizer',
    'create_spoofer',
    'list_available_detectors',
    # Detection models
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
    'AttributeResult',
    'create_attribute_predictor',
    'Emotion',
    'EmotionResult',
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
