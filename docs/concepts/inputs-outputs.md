# Inputs & Outputs

This page describes the data types used throughout UniFace.

---

## Input: Images

All models accept NumPy arrays in **BGR format** (OpenCV default):

```python
import cv2

# Load image (BGR format)
image = cv2.imread("photo.jpg")
print(f"Shape: {image.shape}")  # (H, W, 3)
print(f"Dtype: {image.dtype}")  # uint8
```

!!! warning "Color Format"
    UniFace expects **BGR** format (OpenCV default). If using PIL or other libraries, convert first:

    ```python
    from PIL import Image
    import numpy as np

    pil_image = Image.open("photo.jpg")
    bgr_image = np.array(pil_image)[:, :, ::-1]  # RGB → BGR
    ```

!!! warning "Dtype"
    Images must be `uint8` in `[0, 255]`, three-channel. A float `[0, 1]` array or a
    grayscale frame raises `ValueError` rather than returning quietly wrong results:

    ```python
    detector.detect(image.astype(np.float32) / 255.0)
    # ValueError: Expected dtype uint8, got float32. Scale to [0, 255] and cast with .astype(np.uint8).

    detector.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    # ValueError: Expected a BGR image of shape (H, W, 3), got (480, 640). Convert with cv2.cvtColor.
    ```

    Normalization is each model's job — pass the raw `cv2.imread` array through.

---

## Output: Face Dataclass

Detection returns a list of `Face` objects:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class Face:
    # Required (from detection)
    bbox: np.ndarray        # [x1, y1, x2, y2]
    confidence: float       # 0.0 to 1.0
    landmarks: np.ndarray   # (5, 2) from detectors — except BlazeFace, which returns (6, 2).
                            # Dense landmarkers return (106, 2), (98, 2), (68, 2), (468, 3), or (478, 3).

    # Optional (enriched by analyzers)
    embedding: np.ndarray | None = None
    gender: int | None = None           # 0=Female, 1=Male
    age: int | None = None              # Years
    age_group: str | None = None        # "20-29", etc.
    race: str | None = None             # "East Asian", etc.
    emotion: str | None = None          # "Happy", etc.
    emotion_confidence: float | None = None
    left_eye_open: float | None = None   # [0, 1] probability, from FaceAttribNet
    right_eye_open: float | None = None  # [0, 1] probability, from FaceAttribNet
    eyeglasses: float | None = None      # [0, 1] probability, from FaceAttribNet
    mask: float | None = None            # [0, 1] probability, from FaceAttribNet
    sunglasses: float | None = None      # [0, 1] probability, from FaceAttribNet
    quality: float | None = None        # [0, 1] quality score from eDifFIQA
    track_id: int | None = None         # Persistent ID from tracker
```

### Properties

```python
face = faces[0]

# Bounding box formats
face.bbox_xyxy  # [x1, y1, x2, y2] - same as bbox
face.bbox_xywh  # [x1, y1, width, height]

# Gender as string
face.sex  # "Female" or "Male" (None if not predicted)
```

### Methods

```python
# Compute similarity with another face
similarity = face1.compute_similarity(face2)

# Convert to dictionary
face_dict = face.to_dict()

# Convert to JSON string
face_json = face.to_json(indent=2)
```

---

## Result Types

### GazeResult

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class GazeResult:
    pitch: float  # Vertical angle (radians), + = up
    yaw: float    # Horizontal angle (radians), + = right
```

**Usage:**

```python
import numpy as np

result = gaze_estimator.estimate(face_crop)
print(f"Pitch: {np.degrees(result.pitch):.1f}°")
print(f"Yaw: {np.degrees(result.yaw):.1f}°")
```

---

### HeadPoseResult

```python
@dataclass(frozen=True)
class HeadPoseResult:
    pitch: float  # Rotation around X-axis (degrees), + = looking down
    yaw: float    # Rotation around Y-axis (degrees), + = looking right
    roll: float   # Rotation around Z-axis (degrees), + = tilting clockwise
```

**Usage:**

```python
result = head_pose.estimate(face_crop)
print(f"Pitch: {result.pitch:.1f}°")
print(f"Yaw: {result.yaw:.1f}°")
print(f"Roll: {result.roll:.1f}°")
```

---

### SpoofingResult

```python
@dataclass(frozen=True)
class SpoofingResult:
    is_real: bool      # True = real, False = fake
    confidence: float  # 0.0 to 1.0
```

**Usage:**

```python
result = spoofer.predict(image, face.bbox)
label = "Real" if result.is_real else "Fake"
print(f"{label}: {result.confidence:.1%}")
```

---

### DemographyResult

```python
@dataclass(frozen=True)
class DemographyResult:
    gender: int              # 0=Female, 1=Male
    age: int | None          # Years (AgeGender model)
    age_group: str | None    # "20-29" (FairFace model)
    race: str | None         # Race label (FairFace model)

    @property
    def sex(self) -> str:
        return "Female" if self.gender == 0 else "Male"
```

**Usage:**

```python
# AgeGender model
result = age_gender.predict(image, face)
print(f"{result.sex}, {result.age} years old")

# FairFace model
result = fairface.predict(image, face)
print(f"{result.sex}, {result.age_group}, {result.race}")
```

---

### EmotionResult

```python
@dataclass(frozen=True)
class EmotionResult:
    emotion: str       # "Happy", "Sad", etc.
    confidence: float  # 0.0 to 1.0
```

---

### QualityResult

```python
@dataclass(frozen=True)
class QualityResult:
    score: float  # 0.0 to 1.0, higher = better quality
```

---

### FaceStateResult

```python
@dataclass(frozen=True)
class FaceStateResult:
    left_eye_open: float   # Probability the left eye is open
    right_eye_open: float  # Probability the right eye is open
    eyeglasses: float      # Probability eyeglasses are present
    mask: float            # Probability a face mask is present
    sunglasses: float      # Probability sunglasses are present
```

The five values come from independent binary heads: they do not sum to 1 and several can
be high at once. Threshold each attribute separately; never `argmax`.

**Usage:**

```python
result = face_attrib_net.predict(image, face)
print(result.as_dict())          # {'left_eye_open': 0.98, ...}
print(result.labels(0.5))        # ['left_eye_open', 'right_eye_open', 'eyeglasses']
```

---

### FaceMeshResult

```python
@dataclass(frozen=True)
class FaceMeshResult:
    landmarks: np.ndarray  # (468, 3) or (478, 3); x, y in image pixels, z is relative depth
    score: float           # Face presence, 0.0 to 1.0

    @property
    def points_2d(self) -> np.ndarray:
        return self.landmarks[:, :2]  # (N, 2), depth dropped
```

`score` saturates near 1.0 for anything plausible. It confirms the model ran; it is not a
discriminative confidence, so do not threshold on it.

---

## Embeddings

Face recognition models return normalized 512-dimensional embeddings:

```python
embedding = recognizer.get_normalized_embedding(image, landmarks)
print(f"Shape: {embedding.shape}")  # (512,)
print(f"Norm: {np.linalg.norm(embedding):.4f}")  # ~1.0
```

### Similarity Computation

```python
from uniface.face_utils import compute_similarity

similarity = compute_similarity(embedding1, embedding2)
# Returns: float between -1 and 1 (cosine similarity)
```

---

## Parsing Masks

Face parsing returns a segmentation mask:

```python
mask = parser.parse(face_image)
print(f"Shape: {mask.shape}")  # (H, W)
print(f"Classes: {np.unique(mask)}")  # [0, 1, 2, ...]
```

**19 Classes:**

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | Background | 10 | Nose |
| 1 | Skin | 11 | Mouth |
| 2 | Left Eyebrow | 12 | Upper Lip |
| 3 | Right Eyebrow | 13 | Lower Lip |
| 4 | Left Eye | 14 | Neck |
| 5 | Right Eye | 15 | Necklace |
| 6 | Eyeglasses | 16 | Cloth |
| 7 | Left Ear | 17 | Hair |
| 8 | Right Ear | 18 | Hat |
| 9 | Earring | | |

---

## Next Steps

- [Coordinate Systems](coordinate-systems.md) - Bbox and landmark formats
- [Thresholds & Calibration](thresholds-calibration.md) - Tuning confidence thresholds
