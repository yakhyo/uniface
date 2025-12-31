# Detection

Face detection is the first step in any face analysis pipeline. UniFace provides three detection models.

---

## Available Models

| Model | Backbone | Size | WIDER FACE (Easy/Medium/Hard) | Best For |
|-------|----------|------|-------------------------------|----------|
| **RetinaFace** | MobileNet V2 | 3.5 MB | 91.7% / 91.0% / 86.6% | Balanced (recommended) |
| **SCRFD** | SCRFD-10G | 17 MB | 95.2% / 93.9% / 83.1% | High accuracy |
| **YOLOv5-Face** | YOLOv5s | 28 MB | 94.3% / 92.6% / 83.2% | Real-time |

---

## RetinaFace

The recommended detector for most use cases.

### Basic Usage

```python
from uniface import RetinaFace

detector = RetinaFace()
faces = detector.detect(image)

for face in faces:
    print(f"Confidence: {face.confidence:.2f}")
    print(f"BBox: {face.bbox}")
    print(f"Landmarks: {face.landmarks.shape}")  # (5, 2)
```

### Model Variants

```python
from uniface import RetinaFace
from uniface.constants import RetinaFaceWeights

# Lightweight (mobile/edge)
detector = RetinaFace(model_name=RetinaFaceWeights.MNET_025)

# Balanced (default)
detector = RetinaFace(model_name=RetinaFaceWeights.MNET_V2)

# High accuracy
detector = RetinaFace(model_name=RetinaFaceWeights.RESNET34)
```

| Variant | Params | Size | Easy | Medium | Hard |
|---------|--------|------|------|--------|------|
| MNET_025 | 0.4M | 1.7 MB | 88.5% | 87.0% | 80.6% |
| MNET_050 | 1.0M | 2.6 MB | 89.4% | 88.0% | 82.4% |
| MNET_V1 | 3.5M | 3.8 MB | 90.6% | 89.1% | 84.1% |
| **MNET_V2** :material-check-circle: | 3.2M | 3.5 MB | 91.7% | 91.0% | 86.6% |
| RESNET18 | 11.7M | 27 MB | 92.5% | 91.0% | 86.6% |
| RESNET34 | 24.8M | 56 MB | 94.2% | 93.1% | 88.9% |

### Configuration

```python
detector = RetinaFace(
    model_name=RetinaFaceWeights.MNET_V2,
    confidence_threshold=0.5,  # Min confidence
    nms_threshold=0.4,         # NMS IoU threshold
    input_size=(640, 640),     # Input resolution
    dynamic_size=False         # Enable dynamic input size
)
```

---

## SCRFD

State-of-the-art detection with excellent accuracy-speed tradeoff.

### Basic Usage

```python
from uniface import SCRFD

detector = SCRFD()
faces = detector.detect(image)
```

### Model Variants

```python
from uniface import SCRFD
from uniface.constants import SCRFDWeights

# Real-time (lightweight)
detector = SCRFD(model_name=SCRFDWeights.SCRFD_500M_KPS)

# High accuracy (default)
detector = SCRFD(model_name=SCRFDWeights.SCRFD_10G_KPS)
```

| Variant | Params | Size | Easy | Medium | Hard |
|---------|--------|------|------|--------|------|
| SCRFD_500M_KPS | 0.6M | 2.5 MB | 90.6% | 88.1% | 68.5% |
| **SCRFD_10G_KPS** :material-check-circle: | 4.2M | 17 MB | 95.2% | 93.9% | 83.1% |

### Configuration

```python
detector = SCRFD(
    model_name=SCRFDWeights.SCRFD_10G_KPS,
    confidence_threshold=0.5,
    nms_threshold=0.4,
    input_size=(640, 640)
)
```

---

## YOLOv5-Face

YOLO-based detection optimized for faces.

### Basic Usage

```python
from uniface import YOLOv5Face

detector = YOLOv5Face()
faces = detector.detect(image)
```

### Model Variants

```python
from uniface import YOLOv5Face
from uniface.constants import YOLOv5FaceWeights

# Lightweight
detector = YOLOv5Face(model_name=YOLOv5FaceWeights.YOLOV5N)

# Balanced (default)
detector = YOLOv5Face(model_name=YOLOv5FaceWeights.YOLOV5S)

# High accuracy
detector = YOLOv5Face(model_name=YOLOv5FaceWeights.YOLOV5M)
```

| Variant | Size | Easy | Medium | Hard |
|---------|------|------|--------|------|
| YOLOV5N | 11 MB | 93.6% | 91.5% | 80.5% |
| **YOLOV5S** :material-check-circle: | 28 MB | 94.3% | 92.6% | 83.2% |
| YOLOV5M | 82 MB | 95.3% | 93.8% | 85.3% |

!!! note "Fixed Input Size"
    YOLOv5-Face uses a fixed input size of 640×640.

### Configuration

```python
detector = YOLOv5Face(
    model_name=YOLOv5FaceWeights.YOLOV5S,
    confidence_threshold=0.6,
    nms_threshold=0.5
)
```

---

## Factory Function

Create detectors dynamically:

```python
from uniface import create_detector

detector = create_detector('retinaface')
# or
detector = create_detector('scrfd')
# or
detector = create_detector('yolov5face')
```

---

## High-Level API

One-line detection:

```python
from uniface import detect_faces

faces = detect_faces(
    image,
    method='retinaface',
    confidence_threshold=0.5
)
```

---

## Output Format

All detectors return `list[Face]`:

```python
for face in faces:
    # Bounding box [x1, y1, x2, y2]
    bbox = face.bbox

    # Detection confidence (0-1)
    confidence = face.confidence

    # 5-point landmarks (5, 2)
    landmarks = face.landmarks
    # [left_eye, right_eye, nose, left_mouth, right_mouth]
```

---

## Visualization

```python
from uniface.visualization import draw_detections

draw_detections(
    image=image,
    bboxes=[f.bbox for f in faces],
    scores=[f.confidence for f in faces],
    landmarks=[f.landmarks for f in faces],
    vis_threshold=0.6
)

cv2.imwrite("result.jpg", image)
```

---

## Performance Comparison

Benchmark on your hardware:

```bash
python tools/detection.py --source image.jpg --iterations 100
```

---

## See Also

- [Recognition Module](recognition.md) - Extract embeddings from detected faces
- [Landmarks Module](landmarks.md) - Get 106-point landmarks
- [Image Pipeline Recipe](../recipes/image-pipeline.md) - Complete detection workflow
- [Concepts: Thresholds](../concepts/thresholds-calibration.md) - Tuning detection parameters
