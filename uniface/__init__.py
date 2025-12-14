# Copyright 2025 Yakhyokhuja Valikhujaev
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

__license__ = 'MIT'
__author__ = 'Yakhyokhuja Valikhujaev'
__version__ = '1.5.0'


from uniface.face_utils import compute_similarity, face_alignment
from uniface.log import Logger, enable_logging
from uniface.model_store import verify_model_weights
from uniface.visualization import draw_detections, vis_parsing_maps

from .analyzer import FaceAnalyzer
from .attribute import AgeGender
from .face import Face

try:
    from .attribute import Emotion
except ImportError:
    Emotion = None  # PyTorch not installed
from .detection import (
    SCRFD,
    RetinaFace,
    YOLOv5Face,
    create_detector,
    detect_faces,
    list_available_detectors,
)
from .gaze import MobileGaze, create_gaze_estimator
from .landmark import Landmark106, create_landmarker
from .parsing import BiSeNet, create_face_parser
from .recognition import ArcFace, MobileFace, SphereFace, create_recognizer

__all__ = [
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
    'create_landmarker',
    'create_recognizer',
    'detect_faces',
    'list_available_detectors',
    # Detection models
    'RetinaFace',
    'SCRFD',
    'YOLOv5Face',
    # Recognition models
    'ArcFace',
    'MobileFace',
    'SphereFace',
    # Landmark models
    'Landmark106',
    # Gaze models
    'MobileGaze',
    # Parsing models
    'BiSeNet',
    # Attribute models
    'AgeGender',
    'Emotion',
    # Utilities
    'compute_similarity',
    'draw_detections',
    'vis_parsing_maps',
    'face_alignment',
    'verify_model_weights',
    'Logger',
    'enable_logging',
]
