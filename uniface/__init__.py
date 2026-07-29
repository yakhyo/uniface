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
__version__ = '3.7.1'

import contextlib

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
