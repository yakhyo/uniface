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

__license__ = "MIT"
__author__ = "Yakhyokhuja Valikhujaev"
__version__ = "0.1.8"


from .detection import detect_faces, create_detector, list_available_detectors
from .recognition import create_recognizer
from .landmark import create_landmarker

from uniface.face_utils import face_alignment, compute_similarity
from uniface.model_store import verify_model_weights
from uniface.visualization import draw_detections

from uniface.log import Logger


__all__ = [
    '__author__',
    '__license__',
    '__version__',

    'create_detector',
    'create_landmarker',
    'create_recognizer',
    'detect_faces',
    'list_available_detectors',

    'compute_similarity',
    'draw_detections',
    'face_alignment',
    'verify_model_weights',

    'Logger'
]
