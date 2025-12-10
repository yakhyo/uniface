# MLX-UniFace API Reference

Complete API documentation for MLX-UniFace v1.3.1.

---

## Table of Contents

1. [Core Classes](#core-classes)
   - [Face](#face)
   - [FaceAnalyzer](#faceanalyzer)
2. [Detection](#detection)
   - [Factory Functions](#detection-factory-functions)
   - [Detectors](#detectors)
   - [Weight Enums](#detection-weight-enums)
3. [Recognition](#recognition)
   - [Factory Functions](#recognition-factory-functions)
   - [Recognizers](#recognizers)
   - [Weight Enums](#recognition-weight-enums)
4. [Landmarks](#landmarks)
5. [Attributes](#attributes)
6. [Utilities](#utilities)
7. [Backend Configuration](#backend-configuration)

---

## Core Classes

### Face

```python
from uniface import Face
```

A dataclass representing a detected face with analysis results.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `bbox` | `np.ndarray` | Bounding box coordinates `[x1, y1, x2, y2]` |
| `confidence` | `float` | Detection confidence score (0.0-1.0) |
| `landmarks` | `np.ndarray` | Facial landmarks, shape `(5, 2)` for 5-point |
| `embedding` | `Optional[np.ndarray]` | 512-dimensional face embedding vector |
| `age` | `Optional[int]` | Predicted age |
| `gender` | `Optional[int]` | Predicted gender (0=Female, 1=Male) |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `sex` | `str` | Gender as string: `"Female"` or `"Male"` |
| `bbox_xyxy` | `np.ndarray` | Bounding box in `(x1, y1, x2, y2)` format |
| `bbox_xywh` | `np.ndarray` | Bounding box in `(x, y, width, height)` format |

#### Methods

##### `compute_similarity(other: Face) -> float`

Compute cosine similarity between this face and another face.

```python
similarity = face1.compute_similarity(face2)
print(f"Similarity: {similarity:.4f}")  # 0.0 to 1.0
```

**Raises:** `ValueError` if either face lacks an embedding.

##### `to_dict() -> dict`

Convert the Face object to a dictionary.

```python
face_dict = face.to_dict()
```

---

### FaceAnalyzer

```python
from uniface import FaceAnalyzer, create_detector, create_recognizer, AgeGender
```

High-level orchestrator combining detection, recognition, and attribute analysis.

#### Constructor

```python
FaceAnalyzer(
    detector: BaseDetector,
    recognizer: Optional[BaseRecognizer] = None,
    age_gender: Optional[AgeGender] = None
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `detector` | `BaseDetector` | Face detector instance (required) |
| `recognizer` | `BaseRecognizer` | Face recognizer for embeddings (optional) |
| `age_gender` | `AgeGender` | Age/gender predictor (optional) |

#### Methods

##### `analyze(image: np.ndarray) -> List[Face]`

Perform complete face analysis on an image.

```python
import cv2
from uniface import FaceAnalyzer, create_detector, create_recognizer, AgeGender

# Initialize components
detector = create_detector('retinaface')
recognizer = create_recognizer('arcface')
age_gender = AgeGender()

# Create analyzer
analyzer = FaceAnalyzer(detector, recognizer, age_gender)

# Analyze image
image = cv2.imread('photo.jpg')
faces = analyzer.analyze(image)

for face in faces:
    print(f"Confidence: {face.confidence:.3f}")
    print(f"Age: {face.age}, Gender: {face.sex}")
    print(f"Embedding shape: {face.embedding.shape}")
```

**Returns:** List of `Face` objects with all available attributes populated.

---

## Detection

### Detection Factory Functions

#### `detect_faces`

```python
from uniface import detect_faces
```

High-level convenience function for face detection with automatic caching.

```python
detect_faces(
    image: np.ndarray,
    method: str = 'retinaface',
    **kwargs
) -> List[Dict[str, Any]]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `np.ndarray` | - | Input image (BGR or RGB) |
| `method` | `str` | `'retinaface'` | Detection method |
| `**kwargs` | - | - | Detector-specific parameters |

**Methods:** `'retinaface'`, `'scrfd'`, `'yolov5face'`

**Returns:** List of dictionaries:
```python
{
    'bbox': np.ndarray,      # [x1, y1, x2, y2]
    'confidence': float,      # 0.0-1.0
    'landmarks': np.ndarray   # (5, 2)
}
```

**Example:**
```python
import cv2
from uniface import detect_faces

image = cv2.imread('photo.jpg')
faces = detect_faces(image, method='retinaface', conf_thresh=0.8)

for face in faces:
    print(f"Found face with confidence: {face['confidence']:.3f}")
    print(f"BBox: {face['bbox']}")
```

---

#### `create_detector`

```python
from uniface import create_detector
```

Factory function to create detector instances.

```python
create_detector(
    method: str = 'retinaface',
    **kwargs
) -> BaseDetector
```

**Example:**
```python
from uniface import create_detector
from uniface.constants import RetinaFaceWeights

detector = create_detector(
    'retinaface',
    model_name=RetinaFaceWeights.MNET_V2,
    conf_thresh=0.7,
    nms_thresh=0.4,
    input_size=(640, 640)
)
```

---

#### `list_available_detectors`

```python
from uniface import list_available_detectors

detectors = list_available_detectors()
for name, info in detectors.items():
    print(f"{name}: {info['description']}")
```

---

### Detectors

#### RetinaFace

State-of-the-art face detector with excellent accuracy.

```python
from uniface import RetinaFace
from uniface.constants import RetinaFaceWeights
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `RetinaFaceWeights` | `MNET_V2` | Model variant |
| `conf_thresh` | `float` | `0.5` | Confidence threshold |
| `nms_thresh` | `float` | `0.4` | NMS IoU threshold |
| `input_size` | `Tuple[int, int]` | `(640, 640)` | Input resolution |
| `pre_nms_topk` | `int` | `5000` | Top-k before NMS |
| `post_nms_topk` | `int` | `750` | Max detections after NMS |
| `dynamic_size` | `bool` | `False` | Dynamic input sizing |

**Example:**
```python
detector = RetinaFace(
    model_name=RetinaFaceWeights.RESNET34,
    conf_thresh=0.6
)
faces = detector.detect(image)
```

---

#### SCRFD

Fast and efficient face detector from InsightFace.

```python
from uniface import SCRFD
from uniface.constants import SCRFDWeights
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `SCRFDWeights` | `SCRFD_10G_KPS` | Model variant |
| `conf_thresh` | `float` | `0.5` | Confidence threshold |
| `nms_thresh` | `float` | `0.4` | NMS IoU threshold |
| `input_size` | `Tuple[int, int]` | `(640, 640)` | Input resolution |

**Example:**
```python
detector = SCRFD(model_name=SCRFDWeights.SCRFD_500M_KPS)
faces = detector.detect(image)
```

---

#### YOLOv5Face

YOLO-based face detector with good speed-accuracy tradeoff.

```python
from uniface import YOLOv5Face
from uniface.constants import YOLOv5FaceWeights
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `YOLOv5FaceWeights` | `YOLOV5S` | Model variant |
| `conf_thresh` | `float` | `0.6` | Confidence threshold |
| `nms_thresh` | `float` | `0.5` | NMS IoU threshold |
| `input_size` | `int` | `640` | Input size (fixed) |
| `max_det` | `int` | `750` | Maximum detections |

**Example:**
```python
detector = YOLOv5Face(model_name=YOLOv5FaceWeights.YOLOV5M)
faces = detector.detect(image)
```

---

### Detection Weight Enums

#### RetinaFaceWeights

| Enum Value | Description | Size |
|------------|-------------|------|
| `MNET_025` | MobileNetV1 width 0.25 | Smallest |
| `MNET_050` | MobileNetV1 width 0.50 | Small |
| `MNET_V1` | MobileNetV1 | Medium |
| `MNET_V2` | MobileNetV2 (recommended) | Medium |
| `RESNET18` | ResNet-18 backbone | Large |
| `RESNET34` | ResNet-34 backbone | Largest |

#### SCRFDWeights

| Enum Value | Description |
|------------|-------------|
| `SCRFD_10G_KPS` | 10 GFLOPs model with keypoints |
| `SCRFD_500M_KPS` | 500M params lightweight model |

#### YOLOv5FaceWeights

| Enum Value | Params | Performance (Easy/Med/Hard) |
|------------|--------|----------------------------|
| `YOLOV5S` | 7.1M | 94.33% / 92.61% / 83.15% |
| `YOLOV5M` | 21.1M | 95.30% / 93.76% / 85.28% |

---

## Recognition

### Recognition Factory Functions

#### `create_recognizer`

```python
from uniface import create_recognizer
```

Factory function to create face recognizer instances.

```python
create_recognizer(
    method: str = 'arcface',
    **kwargs
) -> BaseRecognizer
```

**Methods:** `'arcface'`, `'mobileface'`, `'sphereface'`

**Example:**
```python
from uniface import create_recognizer
from uniface.constants import ArcFaceWeights

recognizer = create_recognizer(
    'arcface',
    model_name=ArcFaceWeights.RESNET
)
```

---

### Recognizers

All recognizers produce 512-dimensional embeddings and accept 112x112 aligned face images.

#### ArcFace

State-of-the-art face recognition using additive angular margin loss.

```python
from uniface import ArcFace
from uniface.constants import ArcFaceWeights
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `ArcFaceWeights` | `MNET` | Model variant |

#### MobileFace

Lightweight face recognition optimized for edge devices.

```python
from uniface import MobileFace
from uniface.constants import MobileFaceWeights
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `MobileFaceWeights` | `MNET_V2` | Model variant |

#### SphereFace

Face recognition using angular softmax loss.

```python
from uniface import SphereFace
from uniface.constants import SphereFaceWeights
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `SphereFaceWeights` | `SPHERE20` | Model variant |

---

### Recognizer Methods

All recognizers implement:

##### `get_embedding(image: np.ndarray, landmarks: Optional[np.ndarray] = None) -> np.ndarray`

Get raw 512-dimensional embedding.

##### `get_normalized_embedding(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray`

Get L2-normalized embedding (recommended for similarity comparison).

```python
from uniface import create_detector, create_recognizer

detector = create_detector('retinaface')
recognizer = create_recognizer('arcface')

faces = detector.detect(image)
embedding = recognizer.get_normalized_embedding(image, faces[0]['landmarks'])
print(f"Embedding shape: {embedding.shape}")  # (512,)
```

---

### Recognition Weight Enums

#### ArcFaceWeights

| Enum Value | Backbone | Description |
|------------|----------|-------------|
| `MNET` | MobileNet | Lightweight, fast |
| `RESNET` | ResNet-50 | High accuracy |

#### MobileFaceWeights

| Enum Value | Description |
|------------|-------------|
| `MNET_025` | Width multiplier 0.25 (ultra-light) |
| `MNET_V2` | MobileNetV2 backbone |
| `MNET_V3_SMALL` | MobileNetV3 small |
| `MNET_V3_LARGE` | MobileNetV3 large |

#### SphereFaceWeights

| Enum Value | Layers | Description |
|------------|--------|-------------|
| `SPHERE20` | 20 | Standard model |
| `SPHERE36` | 36 | Deeper model |

---

## Landmarks

### `create_landmarker`

```python
from uniface import create_landmarker
```

Create a 106-point facial landmark detector.

```python
create_landmarker(method: str = '2d106det', **kwargs) -> BaseLandmarker
```

**Example:**
```python
landmarker = create_landmarker()
landmarks_106 = landmarker.get_landmarks(image, bbox)
print(f"Shape: {landmarks_106.shape}")  # (106, 2)
```

### Landmark106

```python
from uniface import Landmark106
```

Predicts 106 facial landmarks for detailed face geometry.

**Method:**
```python
get_landmarks(image: np.ndarray, bbox: np.ndarray) -> np.ndarray
```

---

## Attributes

### AgeGender

```python
from uniface import AgeGender
```

Predict age and gender from face regions.

**Constructor:**
```python
AgeGender(model_name: AgeGenderWeights = AgeGenderWeights.DEFAULT)
```

**Method:**
```python
predict(image: np.ndarray, bbox: np.ndarray) -> Tuple[int, int]
```

**Returns:** `(gender, age)` where gender is 0 (Female) or 1 (Male)

**Example:**
```python
from uniface import AgeGender

age_gender = AgeGender()
gender, age = age_gender.predict(image, face['bbox'])
print(f"Age: {age}, Gender: {'Female' if gender == 0 else 'Male'}")
```

---

### Emotion (Optional)

```python
from uniface import Emotion  # Requires PyTorch
```

Predict emotions from face regions. Requires PyTorch installation.

**Emotions:** Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt

---

## Utilities

### `compute_similarity`

```python
from uniface import compute_similarity
```

Compute cosine similarity between two face embeddings.

```python
compute_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray
) -> float
```

**Example:**
```python
similarity = compute_similarity(face1_embedding, face2_embedding)
is_same_person = similarity > 0.5  # Typical threshold
```

---

### `face_alignment`

```python
from uniface import face_alignment
```

Align a face using 5-point landmarks.

```python
face_alignment(
    image: np.ndarray,
    landmarks: np.ndarray,
    output_size: Tuple[int, int] = (112, 112)
) -> np.ndarray
```

---

### `draw_detections`

```python
from uniface import draw_detections
```

Visualize detection results on an image.

```python
draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    draw_landmarks: bool = True
) -> np.ndarray
```

**Example:**
```python
import cv2
from uniface import detect_faces, draw_detections

image = cv2.imread('photo.jpg')
faces = detect_faces(image)
result = draw_detections(image, faces)
cv2.imwrite('result.jpg', result)
```

---

### `verify_model_weights`

```python
from uniface import verify_model_weights
```

Download and verify model weights with SHA-256 hash checking.

```python
verify_model_weights(
    model_name: Enum,
    download_dir: Optional[str] = None
) -> str
```

**Returns:** Path to the verified model file.

---

### Logging

```python
from uniface import enable_logging, Logger
import logging

# Enable info-level logging
enable_logging(level=logging.INFO)

# Or debug for more verbose output
enable_logging(level=logging.DEBUG)
```

---

## Backend Configuration

MLX-UniFace supports dual backends: MLX (Apple Silicon) and ONNX (cross-platform).

### Backend Selection

```python
from uniface.backend import (
    Backend,
    get_backend,
    set_backend,
    get_available_backends,
    use_mlx,
    use_onnx
)
```

#### `set_backend`

```python
set_backend(backend: Backend) -> None
```

Set the computation backend.

```python
from uniface.backend import Backend, set_backend

# Use MLX on Apple Silicon
set_backend(Backend.MLX)

# Use ONNX for cross-platform
set_backend(Backend.ONNX)

# Auto-select best available
set_backend(Backend.AUTO)
```

#### `get_backend`

```python
current = get_backend()
print(f"Current backend: {current}")
```

#### `get_available_backends`

```python
backends = get_available_backends()
print(f"Available: {backends}")  # ['mlx', 'onnx'] or ['onnx']
```

#### `use_mlx` / `use_onnx`

```python
if use_mlx():
    print("Using MLX backend for Apple Silicon")
else:
    print("Using ONNX backend")
```

---

## Installation Options

```bash
# MLX backend (Apple Silicon M1/M2/M3/M4)
pip install mlx-uniface[mlx]

# ONNX backend (cross-platform)
pip install mlx-uniface[onnx]

# ONNX with CUDA support
pip install mlx-uniface[gpu]

# All backends
pip install mlx-uniface[all]
```

---

## Complete Example

```python
import cv2
from uniface import (
    FaceAnalyzer,
    create_detector,
    create_recognizer,
    AgeGender,
    draw_detections,
    enable_logging
)
import logging

# Enable logging for debugging
enable_logging(level=logging.INFO)

# Create components
detector = create_detector('retinaface', conf_thresh=0.7)
recognizer = create_recognizer('arcface')
age_gender = AgeGender()

# Create analyzer
analyzer = FaceAnalyzer(detector, recognizer, age_gender)

# Load and analyze image
image = cv2.imread('group_photo.jpg')
faces = analyzer.analyze(image)

# Process results
for i, face in enumerate(faces):
    print(f"\nFace {i + 1}:")
    print(f"  Confidence: {face.confidence:.3f}")
    print(f"  BBox: {face.bbox_xywh}")
    print(f"  Age: {face.age}")
    print(f"  Gender: {face.sex}")
    if face.embedding is not None:
        print(f"  Embedding: {face.embedding.shape}")

# Compare two faces
if len(faces) >= 2:
    similarity = faces[0].compute_similarity(faces[1])
    print(f"\nSimilarity between Face 1 and Face 2: {similarity:.4f}")

# Visualize
result = draw_detections(image, [f.to_dict() for f in faces])
cv2.imwrite('analyzed.jpg', result)
```

---

*Generated for MLX-UniFace v1.3.1*
