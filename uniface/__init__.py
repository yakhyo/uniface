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

from uniface.face_utils import face_alignment, compute_similarity
from uniface.model_store import verify_model_weights
from uniface.visualization import draw_detections

from uniface.log import Logger


__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "__license__",

    # Core functions
    'detect_faces',
    'create_detector',
    'list_available_detectors',

    # Utility functions
    "face_alignment",
    "compute_similarity",
    "verify_model_weights",
    "draw_detections",

    # Classes
    "Logger",
]
