# UniFace: All-in-One Face Analysis Library

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/uniface.svg)](https://pypi.org/project/uniface/)
[![CI](https://github.com/yakhyo/uniface/actions/workflows/ci.yml/badge.svg)](https://github.com/yakhyo/uniface/actions)
[![Downloads](https://pepy.tech/badge/uniface)](https://pepy.tech/project/uniface)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-yakhyo%2Funiface-blue.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/yakhyo/uniface)

<div align="center">
    <img src=".github/logos/logo_web.webp" width=75%>
</div>

**UniFace** is a lightweight, production-ready face analysis library built on ONNX Runtime. It provides high-performance face detection, recognition, landmark detection, and attribute analysis with hardware acceleration support across platforms.

---

## Features

- **High-Speed Face Detection**: ONNX-optimized RetinaFace, SCRFD, and YOLOv5-Face models
- **Facial Landmark Detection**: Accurate 106-point landmark localization
- **Face Recognition**: ArcFace, MobileFace, and SphereFace embeddings
- **Gaze Estimation**: Real-time gaze direction prediction with MobileGaze
- **Attribute Analysis**: Age, gender, and emotion detection
- **Face Alignment**: Precise alignment for downstream tasks
- **Hardware Acceleration**: ARM64 optimizations (Apple Silicon), CUDA (NVIDIA), CPU fallback
- **Simple API**: Intuitive factory functions and clean interfaces
- **Production-Ready**: Type hints, comprehensive logging, PEP8 compliant

---

## Installation

### Quick Install (All Platforms)

```bash
pip install uniface
```

### Platform-Specific Installation

#### macOS (Apple Silicon - M1/M2/M3/M4)

For Apple Silicon Macs, the standard installation automatically includes optimized ARM64 support:

```bash
pip install uniface
```

The base `onnxruntime` package (included with uniface) has native Apple Silicon support with ARM64 optimizations built-in since version 1.13+.

#### Linux/Windows with NVIDIA GPU

For CUDA acceleration on NVIDIA GPUs:

```bash
pip install uniface[gpu]
```

**Requirements:**

- CUDA 11.x or 12.x
- cuDNN 8.x
- See [ONNX Runtime GPU requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

#### CPU-Only (All Platforms)

```bash
pip install uniface
```

### Install from Source

```bash
git clone https://github.com/yakhyo/uniface.git
cd uniface
pip install -e .
```

---

## Quick Start

### Face Detection

```python
import cv2
from uniface import RetinaFace

# Initialize detector
detector = RetinaFace()

# Load image
image = cv2.imread("image.jpg")

# Detect faces
faces = detector.detect(image)

# Process results
for face in faces:
    bbox = face['bbox']  # [x1, y1, x2, y2]
    confidence = face['confidence']
    landmarks = face['landmarks']  # 5-point landmarks
    print(f"Face detected with confidence: {confidence:.2f}")
```

### Face Recognition

```python
from uniface import ArcFace, RetinaFace
from uniface import compute_similarity

# Initialize models
detector = RetinaFace()
recognizer = ArcFace()

# Detect and extract embeddings
faces1 = detector.detect(image1)
faces2 = detector.detect(image2)

embedding1 = recognizer.get_normalized_embedding(image1, faces1[0]['landmarks'])
embedding2 = recognizer.get_normalized_embedding(image2, faces2[0]['landmarks'])

# Compare faces
similarity = compute_similarity(embedding1, embedding2)
print(f"Similarity: {similarity:.4f}")
```

### Facial Landmarks

```python
from uniface import RetinaFace, Landmark106

detector = RetinaFace()
landmarker = Landmark106()

faces = detector.detect(image)
landmarks = landmarker.get_landmarks(image, faces[0]['bbox'])
# Returns 106 (x, y) landmark points
```

### Age & Gender Detection

```python
from uniface import RetinaFace, AgeGender

detector = RetinaFace()
age_gender = AgeGender()

faces = detector.detect(image)
gender, age = age_gender.predict(image, faces[0]['bbox'])
gender_str = 'Female' if gender == 0 else 'Male'
print(f"{gender_str}, {age} years old")
```

### Gaze Estimation

```python
from uniface import RetinaFace, MobileGaze
from uniface.visualization import draw_gaze
import numpy as np

detector = RetinaFace()
gaze_estimator = MobileGaze()

faces = detector.detect(image)
for face in faces:
    bbox = face['bbox']
    x1, y1, x2, y2 = map(int, bbox[:4])
    face_crop = image[y1:y2, x1:x2]

    pitch, yaw = gaze_estimator.estimate(face_crop)
    print(f"Gaze: pitch={np.degrees(pitch):.1f}°, yaw={np.degrees(yaw):.1f}°")

    # Visualize
    draw_gaze(image, bbox, pitch, yaw)
```

---

## Documentation

- [**QUICKSTART.md**](QUICKSTART.md) - 5-minute getting started guide
- [**MODELS.md**](MODELS.md) - Model zoo, benchmarks, and selection guide
- [**Examples**](examples/) - Jupyter notebooks with detailed examples

---

## API Overview

### Factory Functions (Recommended)

```python
from uniface.detection import RetinaFace, SCRFD
from uniface.recognition import ArcFace
from uniface.landmark import Landmark106

from uniface.constants import SCRFDWeights

# Create detector with default settings
detector = RetinaFace()

# Create with custom config
detector = SCRFD(
    model_name=SCRFDWeights.SCRFD_10G_KPS, # SCRFDWeights.SCRFD_500M_KPS
    conf_thresh=0.4,
    input_size=(640, 640)
)
# Or with defaults settings: detector = SCRFD()

# Recognition and landmarks
recognizer = ArcFace()
landmarker = Landmark106()
```

### Direct Model Instantiation

```python
from uniface import RetinaFace, SCRFD, YOLOv5Face, ArcFace, MobileFace, SphereFace
from uniface.constants import RetinaFaceWeights, YOLOv5FaceWeights

# Detection
detector = RetinaFace(
    model_name=RetinaFaceWeights.MNET_V2,
    conf_thresh=0.5,
    nms_thresh=0.4
)
# Or detector = RetinaFace()

# YOLOv5-Face detection
detector = YOLOv5Face(
    model_name=YOLOv5FaceWeights.YOLOV5S,
    conf_thresh=0.6,
    nms_thresh=0.5
)
# Or detector = YOLOv5Face

# Recognition
recognizer = ArcFace()  # Uses default weights
recognizer = MobileFace()  # Lightweight alternative
recognizer = SphereFace()  # Angular softmax alternative
```

### High-Level Detection API

```python
from uniface import detect_faces

# One-line face detection
faces = detect_faces(image, method='retinaface', conf_thresh=0.8)  # methods: retinaface, scrfd, yolov5face
```

### Key Parameters (quick reference)

**Detection**

| Class          | Key params (defaults)                                                                                                                | Notes                                          |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------- |
| `RetinaFace` | `model_name=RetinaFaceWeights.MNET_V2`, `conf_thresh=0.5`, `nms_thresh=0.4`, `input_size=(640, 640)`, `dynamic_size=False` | Supports 5-point landmarks                     |
| `SCRFD`      | `model_name=SCRFDWeights.SCRFD_10G_KPS`, `conf_thresh=0.5`, `nms_thresh=0.4`, `input_size=(640, 640)`                        | Supports 5-point landmarks                     |
| `YOLOv5Face` | `model_name=YOLOv5FaceWeights.YOLOV5S`, `conf_thresh=0.6`, `nms_thresh=0.5`, `input_size=640` (fixed)                        | Supports 5-point landmarks; models: YOLOV5N/S/M; `input_size` must be 640 |

**Recognition**

| Class          | Key params (defaults)                     | Notes                                 |
| -------------- | ----------------------------------------- | ------------------------------------- |
| `ArcFace`    | `model_name=ArcFaceWeights.MNET`        | Returns 512-dim normalized embeddings |
| `MobileFace` | `model_name=MobileFaceWeights.MNET_V2`  | Lightweight embeddings                |
| `SphereFace` | `model_name=SphereFaceWeights.SPHERE20` | Angular softmax variant               |

**Landmark & Attributes**

| Class           | Key params (defaults)                                                 | Notes                                   |
| --------------- | --------------------------------------------------------------------- | --------------------------------------- |
| `Landmark106` | No required params                                                    | 106-point landmarks                     |
| `AgeGender`   | `model_name=AgeGenderWeights.DEFAULT`; `input_size` auto-detected | Requires bbox; ONNXRuntime              |
| `Emotion`     | `model_weights=DDAMFNWeights.AFFECNET7`, `input_size=(112, 112)`  | Requires 5-point landmarks; TorchScript |

**Gaze Estimation**

| Class         | Key params (defaults)                      | Notes                                |
| ------------- | ------------------------------------------ | ------------------------------------ |
| `MobileGaze` | `model_name=GazeWeights.RESNET34`       | Returns (pitch, yaw) angles in radians; trained on Gaze360 |

---

## Model Performance

### Face Detection (WIDER FACE Dataset)

| Model              | Easy   | Medium | Hard   | Use Case               |
| ------------------ | ------ | ------ | ------ | ---------------------- |
| retinaface_mnet025 | 88.48% | 87.02% | 80.61% | Mobile/Edge devices    |
| retinaface_mnet_v2 | 91.70% | 91.03% | 86.60% | Balanced (recommended) |
| retinaface_r34     | 94.16% | 93.12% | 88.90% | High accuracy          |
| scrfd_500m         | 90.57% | 88.12% | 68.51% | Real-time applications |
| scrfd_10g          | 95.16% | 93.87% | 83.05% | Best accuracy/speed    |
| yolov5n_face       | 93.61% | 91.52% | 80.53% | Lightweight/Mobile     |
| yolov5s_face       | 94.33% | 92.61% | 83.15% | Real-time + accuracy   |
| yolov5m_face       | 95.30% | 93.76% | 85.28% | High accuracy          |

_Accuracy values from original papers: [RetinaFace](https://arxiv.org/abs/1905.00641), [SCRFD](https://arxiv.org/abs/2105.04714), [YOLOv5-Face](https://arxiv.org/abs/2105.12931)_

**Benchmark on your hardware:**

```bash
python scripts/run_detection.py --image assets/test.jpg --iterations 100
```

See [MODELS.md](MODELS.md) for detailed model information and selection guide.

<div align="center">
    <img src="assets/test_result.png">
</div>

---

## Examples

### Jupyter Notebooks

Interactive examples covering common face analysis tasks:

| Example | Description | Notebook |
|---------|-------------|----------|
| **Face Detection** | Detect faces and facial landmarks | [face_detection.ipynb](examples/face_detection.ipynb) |
| **Face Alignment** | Align and crop faces for recognition | [face_alignment.ipynb](examples/face_alignment.ipynb) |
| **Face Recognition** | Extract face embeddings and compare faces | [face_analyzer.ipynb](examples/face_analyzer.ipynb) |
| **Face Verification** | Compare two faces to verify identity | [face_verification.ipynb](examples/face_verification.ipynb) |
| **Face Search** | Find a person in a group photo | [face_search.ipynb](examples/face_search.ipynb) |
| **Gaze Estimation** | Estimate gaze direction from face images | [gaze_estimation.ipynb](examples/gaze_estimation.ipynb) |

### Webcam Face Detection

```python
import cv2
from uniface import RetinaFace
from uniface.visualization import draw_detections

detector = RetinaFace()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector.detect(frame)

    # Extract data for visualization
    bboxes = [f['bbox'] for f in faces]
    scores = [f['confidence'] for f in faces]
    landmarks = [f['landmarks'] for f in faces]

    draw_detections(
        image=frame,
        bboxes=bboxes,
        scores=scores,
        landmarks=landmarks,
        vis_threshold=0.6,
    )

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Face Search System

```python
import numpy as np
from uniface import RetinaFace, ArcFace

detector = RetinaFace()
recognizer = ArcFace()

# Build face database
database = {}
for person_id, image_path in person_images.items():
    image = cv2.imread(image_path)
    faces = detector.detect(image)
    if faces:
        embedding = recognizer.get_normalized_embedding(
            image, faces[0]['landmarks']
        )
        database[person_id] = embedding

# Search for a face
query_image = cv2.imread("query.jpg")
query_faces = detector.detect(query_image)
if query_faces:
    query_embedding = recognizer.get_normalized_embedding(
        query_image, query_faces[0]['landmarks']
    )

    # Find best match
    best_match = None
    best_similarity = -1

    for person_id, db_embedding in database.items():
        similarity = np.dot(query_embedding, db_embedding.T)[0][0]
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = person_id

    print(f"Best match: {best_match} (similarity: {best_similarity:.4f})")
```

More examples in the [examples/](examples/) directory.

---

## Advanced Configuration

### Custom ONNX Runtime Providers

```python
from uniface.onnx_utils import get_available_providers, create_onnx_session

# Check available providers
providers = get_available_providers()
print(f"Available: {providers}")

# Force CPU-only execution
from uniface import RetinaFace
detector = RetinaFace()
# Internally uses create_onnx_session() which auto-selects best provider
```

### Model Download and Caching

Models are automatically downloaded on first use and cached in `~/.uniface/models/`.

```python
from uniface.model_store import verify_model_weights
from uniface.constants import RetinaFaceWeights

# Manually download and verify a model
model_path = verify_model_weights(
    RetinaFaceWeights.MNET_V2,
    root='./custom_models'  # Custom cache directory
)
```

### Logging Configuration

```python
from uniface import Logger
import logging

# Set logging level
Logger.setLevel(logging.DEBUG)  # DEBUG, INFO, WARNING, ERROR

# Disable logging
Logger.setLevel(logging.CRITICAL)
```

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=uniface --cov-report=html

# Run specific test file
pytest tests/test_retinaface.py -v
```

---

## Development

### Setup Development Environment

```bash
git clone https://github.com/yakhyo/uniface.git
cd uniface

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

### Code Formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Format code
ruff format .

# Check for linting errors
ruff check .

# Auto-fix linting errors
ruff check . --fix
```

Ruff configuration is in `pyproject.toml`. Key settings:

- Line length: 120
- Python target: 3.10+
- Import sorting: `uniface` as first-party

### Project Structure

```
uniface/
├── uniface/
│   ├── detection/       # Face detection models
│   ├── recognition/     # Face recognition models
│   ├── landmark/        # Landmark detection
│   ├── gaze/            # Gaze estimation
│   ├── attribute/       # Age, gender, emotion
│   ├── onnx_utils.py    # ONNX Runtime utilities
│   ├── model_store.py   # Model download & caching
│   └── visualization.py # Drawing utilities
├── tests/               # Unit tests
├── examples/            # Example notebooks
└── scripts/             # Utility scripts
```

---

## References

- **RetinaFace Training**: [yakhyo/retinaface-pytorch](https://github.com/yakhyo/retinaface-pytorch) - PyTorch implementation and training code
- **YOLOv5-Face ONNX**: [yakhyo/yolov5-face-onnx-inference](https://github.com/yakhyo/yolov5-face-onnx-inference) - ONNX inference implementation
- **Face Recognition Training**: [yakhyo/face-recognition](https://github.com/yakhyo/face-recognition) - ArcFace, MobileFace, SphereFace training code
- **Gaze Estimation Training**: [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation) - MobileGaze training code and pretrained weights
- **InsightFace**: [deepinsight/insightface](https://github.com/deepinsight/insightface) - Model architectures and pretrained weights

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/yakhyo/uniface).
