# MLX-UniFace Architecture

This document describes the system architecture, design patterns, and component relationships in MLX-UniFace.

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Module Structure](#module-structure)
3. [Class Hierarchies](#class-hierarchies)
4. [Backend System](#backend-system)
5. [Data Flow](#data-flow)
6. [Model Weight Management](#model-weight-management)
7. [Neural Network Components](#neural-network-components)
8. [Design Patterns](#design-patterns)

---

## High-Level Overview

MLX-UniFace is a production-ready face analysis library with dual backend support (MLX for Apple Silicon, ONNX for cross-platform). The architecture follows a modular design with clear separation of concerns.

### System Architecture Diagram

```mermaid
graph TB
    subgraph "User Interface"
        API[Public API]
        Factory[Factory Functions]
    end

    subgraph "Orchestration"
        Analyzer[FaceAnalyzer]
    end

    subgraph "Core Components"
        Detection[Detection Module]
        Recognition[Recognition Module]
        Landmark[Landmark Module]
        Attribute[Attribute Module]
    end

    subgraph "Backend Layer"
        Backend[Backend Selector]
        MLX[MLX Backend]
        ONNX[ONNX Backend]
    end

    subgraph "Infrastructure"
        ModelStore[Model Store]
        Utils[Utilities]
        NN[Neural Network Components]
    end

    API --> Factory
    Factory --> Analyzer
    Analyzer --> Detection
    Analyzer --> Recognition
    Analyzer --> Attribute

    Detection --> Backend
    Recognition --> Backend
    Landmark --> Backend
    Attribute --> Backend

    Backend --> MLX
    Backend --> ONNX

    MLX --> NN
    MLX --> ModelStore
    ONNX --> ModelStore

    Detection --> Utils
    Recognition --> Utils
```

---

## Module Structure

```
uniface/
├── __init__.py                 # Public API exports
├── analyzer.py                 # FaceAnalyzer orchestrator
├── backend.py                  # Backend selection (MLX/ONNX)
├── face.py                     # Face dataclass
├── constants.py                # Model enums, URLs, SHA-256 hashes
│
├── detection/                  # Face Detection
│   ├── __init__.py            # Factory: create_detector, detect_faces
│   ├── base.py                # BaseDetector abstract class
│   ├── retinaface.py          # RetinaFace (ONNX)
│   ├── retinaface_mlx.py      # RetinaFace (MLX)
│   ├── scrfd.py               # SCRFD (ONNX)
│   ├── scrfd_mlx.py           # SCRFD (MLX)
│   ├── yolov5.py              # YOLOv5Face (ONNX)
│   └── yolov5_mlx.py          # YOLOv5Face (MLX)
│
├── recognition/               # Face Recognition
│   ├── __init__.py            # Factory: create_recognizer
│   ├── base.py                # BaseRecognizer (ONNX)
│   ├── base_mlx.py            # BaseRecognizerMLX
│   ├── models.py              # ArcFace, MobileFace, SphereFace
│   └── models_mlx.py          # MLX implementations
│
├── landmark/                  # Facial Landmarks
│   ├── __init__.py            # Factory: create_landmarker
│   ├── base.py                # BaseLandmarker
│   ├── models.py              # Landmark106 (ONNX)
│   └── models_mlx.py          # Landmark106MLX
│
├── attribute/                 # Face Attributes
│   ├── __init__.py            # Exports
│   ├── base.py                # Attribute base class
│   ├── age_gender.py          # AgeGender (ONNX)
│   ├── age_gender_mlx.py      # AgeGenderMLX
│   ├── emotion.py             # Emotion (PyTorch)
│   └── emotion_mlx.py         # EmotionMLX
│
├── nn/                        # Neural Network Building Blocks (MLX)
│   ├── __init__.py
│   ├── backbone.py            # MobileNetV1, MobileNetV2
│   ├── conv.py                # Convolution layers
│   ├── fpn.py                 # Feature Pyramid Network
│   └── head.py                # Detection heads
│
├── common.py                  # Shared utilities (anchors, NMS)
├── face_utils.py              # Face alignment, similarity
├── mlx_utils.py               # MLX utilities
├── onnx_utils.py              # ONNX utilities
├── model_store.py             # Model downloading/verification
├── log.py                     # Logging configuration
└── visualization.py           # Drawing utilities
```

---

## Class Hierarchies

### Detection Hierarchy

```mermaid
classDiagram
    class BaseDetector {
        <<abstract>>
        +config: dict
        +detect(image) List~Dict~
        +preprocess(image) ndarray
        +postprocess(outputs) Any
        +supports_landmarks: bool
        +get_info() Dict
    }

    class RetinaFace {
        +model_name: RetinaFaceWeights
        +conf_thresh: float
        +nms_thresh: float
        +input_size: tuple
        +detect(image)
    }

    class RetinaFaceMLX {
        +backbone: nn.Module
        +fpn: FPN
        +heads: list
        +detect(image)
    }

    class SCRFD {
        +model_name: SCRFDWeights
        +feature_strides: list
        +detect(image)
    }

    class SCRFDMLX {
        +detect(image)
    }

    class YOLOv5Face {
        +model_name: YOLOv5FaceWeights
        +max_det: int
        +detect(image)
    }

    class YOLOv5FaceMLX {
        +detect(image)
    }

    BaseDetector <|-- RetinaFace
    BaseDetector <|-- RetinaFaceMLX
    BaseDetector <|-- SCRFD
    BaseDetector <|-- SCRFDMLX
    BaseDetector <|-- YOLOv5Face
    BaseDetector <|-- YOLOv5FaceMLX
```

### Recognition Hierarchy

```mermaid
classDiagram
    class BaseRecognizer {
        <<abstract>>
        +model_name: Enum
        +session: ONNXSession
        +get_embedding(image, landmarks) ndarray
        +get_normalized_embedding(image, landmarks) ndarray
        +preprocess(face_img) ndarray
    }

    class BaseRecognizerMLX {
        <<abstract>>
        +model: nn.Module
        +inference(input) mx.array
        +_build_model() nn.Module
    }

    class ArcFace {
        +model_name: ArcFaceWeights
    }

    class ArcFaceMLX {
        +_build_model()
    }

    class MobileFace {
        +model_name: MobileFaceWeights
    }

    class MobileFaceMLX {
        +_build_model()
    }

    class SphereFace {
        +model_name: SphereFaceWeights
    }

    class SphereFaceMLX {
        +_build_model()
    }

    BaseRecognizer <|-- ArcFace
    BaseRecognizer <|-- MobileFace
    BaseRecognizer <|-- SphereFace

    BaseRecognizerMLX <|-- ArcFaceMLX
    BaseRecognizerMLX <|-- MobileFaceMLX
    BaseRecognizerMLX <|-- SphereFaceMLX
```

### Attribute Hierarchy

```mermaid
classDiagram
    class Attribute {
        <<abstract>>
        +_initialize_model()
        +preprocess(image) Any
        +postprocess(prediction) Any
        +predict(image) Any
        +__call__()
    }

    class AgeGender {
        +predict(image, bbox) Tuple
    }

    class AgeGenderMLX {
        +model: nn.Module
        +predict(image, bbox) Tuple
    }

    class Emotion {
        +labels: list
        +predict(image, bbox) str
    }

    class EmotionMLX {
        +predict(image, bbox) str
    }

    Attribute <|-- AgeGender
    Attribute <|-- AgeGenderMLX
    Attribute <|-- Emotion
    Attribute <|-- EmotionMLX
```

---

## Backend System

The backend system provides seamless switching between MLX and ONNX implementations.

### Backend Selection Flow

```mermaid
flowchart TD
    A[Application Start] --> B{Backend Set?}
    B -->|No| C[Auto-detect]
    B -->|Yes| D[Use Specified]

    C --> E{Apple Silicon?}
    E -->|Yes| F{MLX Installed?}
    E -->|No| G[Use ONNX]

    F -->|Yes| H[Use MLX]
    F -->|No| G

    D --> I{Backend Available?}
    I -->|Yes| J[Use Specified]
    I -->|No| K[Raise Error]

    H --> L[Import MLX Models]
    G --> M[Import ONNX Models]
    J --> N{MLX or ONNX?}
    N -->|MLX| L
    N -->|ONNX| M
```

### Backend API

```python
from uniface.backend import Backend, set_backend, get_backend

# Auto-selection (default)
set_backend(Backend.AUTO)

# Force MLX (Apple Silicon)
set_backend(Backend.MLX)

# Force ONNX (cross-platform)
set_backend(Backend.ONNX)
```

---

## Data Flow

### Complete Analysis Pipeline

```mermaid
sequenceDiagram
    participant User
    participant FaceAnalyzer
    participant Detector
    participant Recognizer
    participant AgeGender
    participant Face

    User->>FaceAnalyzer: analyze(image)
    FaceAnalyzer->>Detector: detect(image)
    Detector-->>FaceAnalyzer: List[{bbox, confidence, landmarks}]

    loop For each detection
        FaceAnalyzer->>Recognizer: get_normalized_embedding(image, landmarks)
        Recognizer-->>FaceAnalyzer: embedding (512,)

        FaceAnalyzer->>AgeGender: predict(image, bbox)
        AgeGender-->>FaceAnalyzer: (gender, age)

        FaceAnalyzer->>Face: Create Face object
    end

    FaceAnalyzer-->>User: List[Face]
```

### Detection Pipeline (RetinaFace)

```mermaid
flowchart LR
    A[Input Image] --> B[Resize & Pad]
    B --> C[Normalize]
    C --> D[Backbone]
    D --> E[FPN]
    E --> F[Classification Head]
    E --> G[BBox Head]
    E --> H[Landmark Head]
    F --> I[Decode]
    G --> I
    H --> I
    I --> J[NMS]
    J --> K[Detections]
```

### Recognition Pipeline

```mermaid
flowchart LR
    A[Full Image] --> B[5-Point Landmarks]
    B --> C[Face Alignment]
    C --> D[112x112 Crop]
    D --> E[Normalize]
    E --> F[Encoder Network]
    F --> G[512-dim Embedding]
    G --> H[L2 Normalize]
    H --> I[Final Embedding]
```

---

## Model Weight Management

### Weight Download Flow

```mermaid
flowchart TD
    A[Model Request] --> B{Weights Cached?}
    B -->|Yes| C{Hash Valid?}
    B -->|No| D[Download from GitHub]

    C -->|Yes| E[Load Model]
    C -->|No| D

    D --> F[Save to ~/.uniface/models/]
    F --> G[Compute SHA-256]
    G --> H{Hash Matches?}
    H -->|Yes| E
    H -->|No| I[Delete & Retry]
    I --> D
```

### Model Storage Structure

```
~/.uniface/models/
├── retinaface_mv2.onnx           # ONNX weights
├── retinaface_mnet_v2.safetensors # MLX weights
├── w600k_mbf.onnx                 # ArcFace ONNX
├── arcface_mnet.safetensors       # ArcFace MLX
├── genderage.onnx                 # Age/Gender
├── age_gender.safetensors         # Age/Gender MLX
└── ...
```

---

## Neural Network Components

### RetinaFace Architecture (MLX)

```mermaid
graph TB
    subgraph "Backbone (MobileNetV2)"
        Input[Input 640x640x3]
        Stage1[Stage 1: stride 8]
        Stage2[Stage 2: stride 16]
        Stage3[Stage 3: stride 32]
    end

    subgraph "FPN"
        P3[P3 Features]
        P4[P4 Features]
        P5[P5 Features]
        Lateral3[Lateral Conv]
        Lateral4[Lateral Conv]
        Lateral5[Lateral Conv]
        Up4[Upsample 2x]
        Up5[Upsample 2x]
    end

    subgraph "SSH Context Module"
        SSH3[SSH P3]
        SSH4[SSH P4]
        SSH5[SSH P5]
    end

    subgraph "Detection Heads"
        Cls[Class Head<br/>num_anchors x 2]
        BBox[BBox Head<br/>num_anchors x 4]
        LM[Landmark Head<br/>num_anchors x 10]
    end

    Input --> Stage1 --> Stage2 --> Stage3

    Stage1 --> Lateral3
    Stage2 --> Lateral4
    Stage3 --> Lateral5

    Lateral5 --> Up5
    Up5 --> |+| Lateral4
    Lateral4 --> Up4
    Up4 --> |+| Lateral3

    Lateral3 --> SSH3
    Lateral4 --> SSH4
    Lateral5 --> SSH5

    SSH3 --> Cls
    SSH4 --> Cls
    SSH5 --> Cls

    SSH3 --> BBox
    SSH4 --> BBox
    SSH5 --> BBox

    SSH3 --> LM
    SSH4 --> LM
    SSH5 --> LM
```

### Inverted Residual Block (MobileNetV2)

```mermaid
graph LR
    A[Input] --> B[1x1 Conv Expand]
    B --> C[BatchNorm + ReLU6]
    C --> D[3x3 DWConv]
    D --> E[BatchNorm + ReLU6]
    E --> F[1x1 Conv Project]
    F --> G[BatchNorm]
    A --> H[Residual]
    H --> |+| G
    G --> I[Output]
```

---

## Design Patterns

### 1. Factory Pattern

Factory functions abstract the creation of complex objects:

```python
# Detection factory
detector = create_detector('retinaface', conf_thresh=0.7)

# Recognition factory
recognizer = create_recognizer('arcface')

# Landmark factory
landmarker = create_landmarker('2d106det')
```

### 2. Strategy Pattern

Backend selection allows swapping implementations at runtime:

```python
# Same API, different backends
if use_mlx():
    from .retinaface_mlx import RetinaFaceMLX as RetinaFace
else:
    from .retinaface import RetinaFace
```

### 3. Template Method Pattern

Base classes define the algorithm structure:

```python
class BaseDetector(ABC):
    def detect(self, image):
        preprocessed = self.preprocess(image)  # Step 1
        outputs = self._inference(preprocessed)  # Step 2
        return self.postprocess(outputs)  # Step 3

    @abstractmethod
    def preprocess(self, image): pass

    @abstractmethod
    def postprocess(self, outputs): pass
```

### 4. Singleton Pattern (Caching)

Detector instances are cached for reuse:

```python
_detector_cache: Dict[str, BaseDetector] = {}

def detect_faces(image, method='retinaface', **kwargs):
    cache_key = f'{method}_{sorted(kwargs.items())}'
    if cache_key not in _detector_cache:
        _detector_cache[cache_key] = create_detector(method, **kwargs)
    return _detector_cache[cache_key].detect(image)
```

### 5. Dataclass Pattern

The `Face` class uses Python dataclasses for clean data representation:

```python
@dataclass
class Face:
    bbox: np.ndarray
    confidence: float
    landmarks: np.ndarray
    embedding: Optional[np.ndarray] = None
    age: Optional[int] = None
    gender: Optional[int] = None
```

---

## Hardware Acceleration

### ONNX Provider Selection

```mermaid
flowchart TD
    A[Create Session] --> B{CoreML Available?}
    B -->|Yes| C[Use CoreMLExecutionProvider]
    B -->|No| D{CUDA Available?}
    D -->|Yes| E[Use CUDAExecutionProvider]
    D -->|No| F[Use CPUExecutionProvider]

    C --> G[Session Ready]
    E --> G
    F --> G
```

### MLX Device Selection

```mermaid
flowchart TD
    A[MLX Init] --> B{Apple Silicon?}
    B -->|Yes| C{Metal Available?}
    C -->|Yes| D[GPU Device]
    C -->|No| E[CPU Device]
    B -->|No| F[Error: MLX requires Apple Silicon]
```

---

## Key Architectural Decisions

### 1. Dual Backend Support

**Decision:** Support both MLX and ONNX backends with automatic selection.

**Rationale:**
- MLX provides native Apple Silicon optimization (2-10x faster)
- ONNX ensures cross-platform compatibility
- Auto-selection provides best UX

### 2. Weight Verification

**Decision:** SHA-256 hash verification for all model downloads.

**Rationale:**
- Ensures model integrity
- Prevents corrupted downloads
- Security against tampering

### 3. Lazy Loading

**Decision:** Models are downloaded and loaded on first use.

**Rationale:**
- Faster startup time
- Reduced memory usage
- Only download what's needed

### 4. Unified API

**Decision:** Same API surface for both backends.

**Rationale:**
- Seamless backend switching
- Simplified testing
- Better maintainability

---

*Architecture documentation for MLX-UniFace v1.3.1*
