# API Reference

Quick reference for all UniFace classes and functions.

---

## Detection

### RetinaFace

```python
from uniface import RetinaFace

detector = RetinaFace(
    model_name=RetinaFaceWeights.MNET_V2,  # Model variant
    confidence_threshold=0.5,               # Min confidence
    nms_threshold=0.4,                       # NMS IoU threshold
    input_size=(640, 640)                    # Input resolution
)

faces = detector.detect(image)  # Returns list[Face]
```

### SCRFD

```python
from uniface import SCRFD

detector = SCRFD(
    model_name=SCRFDWeights.SCRFD_10G_KPS,
    confidence_threshold=0.5,
    nms_threshold=0.4,
    input_size=(640, 640)
)
```

### YOLOv5Face

```python
from uniface import YOLOv5Face

detector = YOLOv5Face(
    model_name=YOLOv5FaceWeights.YOLOV5S,
    confidence_threshold=0.6,
    nms_threshold=0.5
)
```

---

## Recognition

### ArcFace

```python
from uniface import ArcFace

recognizer = ArcFace(model_name=ArcFaceWeights.MNET)

embedding = recognizer.get_normalized_embedding(image, landmarks)
# Returns: np.ndarray (1, 512)
```

### MobileFace / SphereFace

```python
from uniface import MobileFace, SphereFace

recognizer = MobileFace(model_name=MobileFaceWeights.MNET_V2)
recognizer = SphereFace(model_name=SphereFaceWeights.SPHERE20)
```

---

## Landmarks

```python
from uniface import Landmark106

landmarker = Landmark106()
landmarks = landmarker.get_landmarks(image, bbox)
# Returns: np.ndarray (106, 2)
```

---

## Attributes

### AgeGender

```python
from uniface import AgeGender

predictor = AgeGender()
result = predictor.predict(image, bbox)
# Returns: AttributeResult(gender, age, sex)
```

### FairFace

```python
from uniface import FairFace

predictor = FairFace()
result = predictor.predict(image, bbox)
# Returns: AttributeResult(gender, age_group, race, sex)
```

---

## Gaze

```python
from uniface import MobileGaze

gaze = MobileGaze(model_name=GazeWeights.RESNET34)
result = gaze.estimate(face_crop)
# Returns: GazeResult(pitch, yaw) in radians
```

---

## Parsing

```python
from uniface.parsing import BiSeNet

parser = BiSeNet(model_name=ParsingWeights.RESNET18)
mask = parser.parse(face_image)
# Returns: np.ndarray (H, W) with values 0-18
```

---

## Anti-Spoofing

```python
from uniface.spoofing import MiniFASNet

spoofer = MiniFASNet(model_name=MiniFASNetWeights.V2)
result = spoofer.predict(image, bbox)
# Returns: SpoofingResult(is_real, confidence)
```

---

## Privacy

```python
from uniface.privacy import BlurFace, anonymize_faces

# One-liner
anonymized = anonymize_faces(image, method='pixelate')

# Manual control
blurrer = BlurFace(method='gaussian', blur_strength=3.0)
anonymized = blurrer.anonymize(image, faces)
```

---

## Types

### Face

```python
@dataclass
class Face:
    bbox: np.ndarray        # [x1, y1, x2, y2]
    confidence: float       # 0.0 to 1.0
    landmarks: np.ndarray   # (5, 2)
    embedding: np.ndarray | None = None
    gender: int | None = None
    age: int | None = None
    age_group: str | None = None
    race: str | None = None
```

### Result Types

```python
GazeResult(pitch: float, yaw: float)
SpoofingResult(is_real: bool, confidence: float)
AttributeResult(gender: int, age: int, age_group: str, race: str)
EmotionResult(emotion: str, confidence: float)
```

---

## Utilities

```python
from uniface import (
    compute_similarity,      # Compare embeddings
    face_alignment,          # Align face for recognition
    draw_detections,         # Visualize detections
    vis_parsing_maps,        # Visualize parsing
    verify_model_weights,    # Download/verify models
)
```
