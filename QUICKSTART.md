# UniFace Quick Start Guide

Get up and running with UniFace in 5 minutes! This guide covers the most common use cases.

---

## Installation

```bash
# macOS (Apple Silicon) - automatically includes ARM64 optimizations
pip install uniface

# Linux/Windows with NVIDIA GPU
pip install uniface[gpu]

# CPU-only (all platforms)
pip install uniface
```

---

## 1. Face Detection (30 seconds)

Detect faces in an image:

```python
import cv2
from uniface import RetinaFace

# Load image
image = cv2.imread("photo.jpg")

# Initialize detector (models auto-download on first use)
detector = RetinaFace()

# Detect faces
faces = detector.detect(image)

# Print results
for i, face in enumerate(faces):
    print(f"Face {i+1}:")
    print(f"  Confidence: {face['confidence']:.2f}")
    print(f"  BBox: {face['bbox']}")
    print(f"  Landmarks: {len(face['landmarks'])} points")
```

**Output:**

```
Face 1:
  Confidence: 0.99
  BBox: [120.5, 85.3, 245.8, 210.6]
  Landmarks: 5 points
```

---

## 2. Visualize Detections (1 minute)

Draw bounding boxes and landmarks:

```python
import cv2
from uniface import RetinaFace
from uniface.visualization import draw_detections

# Detect faces
detector = RetinaFace()
image = cv2.imread("photo.jpg")
faces = detector.detect(image)

# Extract visualization data
bboxes = [f['bbox'] for f in faces]
scores = [f['confidence'] for f in faces]
landmarks = [f['landmarks'] for f in faces]

# Draw on image
draw_detections(image, bboxes, scores, landmarks, vis_threshold=0.6)

# Save result
cv2.imwrite("output.jpg", image)
print("Saved output.jpg")
```

---

## 3. Face Recognition (2 minutes)

Compare two faces:

```python
import cv2
import numpy as np
from uniface import RetinaFace, ArcFace

# Initialize models
detector = RetinaFace()
recognizer = ArcFace()

# Load two images
image1 = cv2.imread("person1.jpg")
image2 = cv2.imread("person2.jpg")

# Detect faces
faces1 = detector.detect(image1)
faces2 = detector.detect(image2)

if faces1 and faces2:
    # Extract embeddings
    emb1 = recognizer.get_normalized_embedding(image1, faces1[0]['landmarks'])
    emb2 = recognizer.get_normalized_embedding(image2, faces2[0]['landmarks'])

    # Compute similarity (cosine similarity)
    similarity = np.dot(emb1, emb2.T)[0][0]

    # Interpret result
    if similarity > 0.6:
        print(f"Same person (similarity: {similarity:.3f})")
    else:
        print(f"Different people (similarity: {similarity:.3f})")
else:
    print("No faces detected")
```

**Similarity thresholds:**

- `> 0.6`: Same person (high confidence)
- `0.4 - 0.6`: Uncertain (manual review)
- `< 0.4`: Different people

---

## 4. Webcam Demo (2 minutes)

Real-time face detection:

```python
import cv2
from uniface import RetinaFace
from uniface.visualization import draw_detections

detector = RetinaFace()
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    faces = detector.detect(frame)

    # Draw results
    bboxes = [f['bbox'] for f in faces]
    scores = [f['confidence'] for f in faces]
    landmarks = [f['landmarks'] for f in faces]
    draw_detections(frame, bboxes, scores, landmarks)

    # Show frame
    cv2.imshow("UniFace - Press 'q' to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 5. Age & Gender Detection (2 minutes)

Detect age and gender:

```python
import cv2
from uniface import RetinaFace, AgeGender

# Initialize models
detector = RetinaFace()
age_gender = AgeGender()

# Load image
image = cv2.imread("photo.jpg")
faces = detector.detect(image)

# Predict attributes
for i, face in enumerate(faces):
    gender_id, age = age_gender.predict(image, face['bbox'])
    gender = 'Female' if gender_id == 0 else 'Male'
    print(f"Face {i+1}: {gender}, {age} years old")
```

**Output:**

```
Face 1: Male, 32 years old
Face 2: Female, 28 years old
```

---

## 6. Facial Landmarks (2 minutes)

Detect 106 facial landmarks:

```python
import cv2
from uniface import RetinaFace, Landmark106

# Initialize models
detector = RetinaFace()
landmarker = Landmark106()

# Detect face and landmarks
image = cv2.imread("photo.jpg")
faces = detector.detect(image)

if faces:
    landmarks = landmarker.get_landmarks(image, faces[0]['bbox'])
    print(f"Detected {len(landmarks)} landmarks")

    # Draw landmarks
    for x, y in landmarks.astype(int):
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    cv2.imwrite("landmarks.jpg", image)
```

---

## 7. Batch Processing (3 minutes)

Process multiple images:

```python
import cv2
from pathlib import Path
from uniface import RetinaFace

detector = RetinaFace()

# Process all images in a folder
image_dir = Path("images/")
output_dir = Path("output/")
output_dir.mkdir(exist_ok=True)

for image_path in image_dir.glob("*.jpg"):
    print(f"Processing {image_path.name}...")

    image = cv2.imread(str(image_path))
    faces = detector.detect(image)

    print(f"  Found {len(faces)} face(s)")

    # Save results
    output_path = output_dir / image_path.name
    # ... draw and save ...

print("Done!")
```

---

## 8. Model Selection

Choose the right model for your use case:

### Detection Models

```python
from uniface.detection import RetinaFace, SCRFD, YOLOv5Face
from uniface.constants import RetinaFaceWeights, SCRFDWeights, YOLOv5FaceWeights

# Fast detection (mobile/edge devices)
detector = RetinaFace(
    model_name=RetinaFaceWeights.MNET_025,
    conf_thresh=0.7
)

# Balanced (recommended)
detector = RetinaFace(
    model_name=RetinaFaceWeights.MNET_V2
)

# Real-time with high accuracy
detector = YOLOv5Face(
    model_name=YOLOv5FaceWeights.YOLOV5S,
    conf_thresh=0.6,
    nms_thresh=0.5
)

# High accuracy (server/GPU)
detector = SCRFD(
    model_name=SCRFDWeights.SCRFD_10G_KPS,
    conf_thresh=0.5
)
```

### Recognition Models

```python
from uniface import ArcFace, MobileFace, SphereFace
from uniface.constants import MobileFaceWeights, SphereFaceWeights

# ArcFace (recommended for most use cases)
recognizer = ArcFace()  # Best accuracy

# MobileFace (lightweight for mobile/edge)
recognizer = MobileFace(model_name=MobileFaceWeights.MNET_V2)  # Fast, small size

# SphereFace (angular margin approach)
recognizer = SphereFace(model_name=SphereFaceWeights.SPHERE20)  # Alternative method
```

---

## Common Issues

### 1. Models Not Downloading

```python
# Manually download a model
from uniface.model_store import verify_model_weights
from uniface.constants import RetinaFaceWeights

model_path = verify_model_weights(RetinaFaceWeights.MNET_V2)
print(f"Model downloaded to: {model_path}")
```

### 2. Check Hardware Acceleration

```python
import onnxruntime as ort
print("Available providers:", ort.get_available_providers())

# macOS M-series should show: ['CoreMLExecutionProvider', ...]
# NVIDIA GPU should show: ['CUDAExecutionProvider', ...]
```

### 3. Slow Performance on Mac

The standard installation includes ARM64 optimizations for Apple Silicon. If performance is slow, verify you're using the ARM64 build of Python:

```bash
python -c "import platform; print(platform.machine())"
# Should show: arm64 (not x86_64)
```

### 4. Import Errors

```python
# Correct imports
from uniface.detection import RetinaFace
from uniface.recognition import ArcFace
from uniface.landmark import Landmark106

# Wrong imports
from uniface import retinaface  # Module, not class
```

---

## Next Steps

- **Detailed Examples**: Check the [examples/](examples/) folder for Jupyter notebooks
- **Model Benchmarks**: See [MODELS.md](MODELS.md) for performance comparisons
- **Full Documentation**: Read [README.md](README.md) for complete API reference

---

## References

- **RetinaFace Training**: [yakhyo/retinaface-pytorch](https://github.com/yakhyo/retinaface-pytorch)
- **YOLOv5-Face Original**: [deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face)
- **YOLOv5-Face ONNX**: [yakhyo/yolov5-face-onnx-inference](https://github.com/yakhyo/yolov5-face-onnx-inference)
- **Face Recognition Training**: [yakhyo/face-recognition](https://github.com/yakhyo/face-recognition)
- **InsightFace**: [deepinsight/insightface](https://github.com/deepinsight/insightface)
