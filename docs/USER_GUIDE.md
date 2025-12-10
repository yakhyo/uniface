# MLX-UniFace User Guide

A comprehensive guide to using MLX-UniFace for face analysis tasks.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Face Detection](#face-detection)
4. [Face Recognition](#face-recognition)
5. [Face Comparison](#face-comparison)
6. [Age & Gender Prediction](#age--gender-prediction)
7. [Complete Analysis Pipeline](#complete-analysis-pipeline)
8. [Working with Landmarks](#working-with-landmarks)
9. [Visualization](#visualization)
10. [Performance Optimization](#performance-optimization)
11. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.10 or higher
- For MLX backend: Apple Silicon Mac (M1/M2/M3/M4)

### Install with pip

```bash
# MLX backend (Apple Silicon - recommended for Mac)
pip install mlx-uniface[mlx]

# ONNX backend (cross-platform)
pip install mlx-uniface[onnx]

# ONNX with CUDA support (NVIDIA GPU)
pip install mlx-uniface[gpu]

# All backends
pip install mlx-uniface[all]

# Development installation
pip install mlx-uniface[dev]
```

### Verify Installation

```python
import uniface
print(f"MLX-UniFace version: {uniface.__version__}")

# Check available backends
from uniface.backend import get_available_backends
print(f"Available backends: {get_available_backends()}")
```

---

## Quick Start

```python
import cv2
from uniface import detect_faces, draw_detections

# Load image
image = cv2.imread('photo.jpg')

# Detect faces
faces = detect_faces(image)
print(f"Found {len(faces)} face(s)")

# Visualize results
result = draw_detections(image, faces)
cv2.imwrite('result.jpg', result)
```

---

## Face Detection

### Basic Detection

```python
from uniface import detect_faces
import cv2

image = cv2.imread('photo.jpg')

# Simple detection with defaults
faces = detect_faces(image)

for face in faces:
    print(f"Confidence: {face['confidence']:.3f}")
    print(f"BBox: {face['bbox']}")  # [x1, y1, x2, y2]
    print(f"Landmarks: {face['landmarks'].shape}")  # (5, 2)
```

### Choosing a Detector

```python
from uniface import detect_faces

# RetinaFace - Best accuracy (default)
faces = detect_faces(image, method='retinaface')

# SCRFD - Fast and efficient
faces = detect_faces(image, method='scrfd')

# YOLOv5Face - Good balance
faces = detect_faces(image, method='yolov5face')
```

### Custom Configuration

```python
from uniface import create_detector
from uniface.constants import RetinaFaceWeights

# High-confidence detection
detector = create_detector(
    'retinaface',
    model_name=RetinaFaceWeights.RESNET34,  # High-accuracy model
    conf_thresh=0.8,  # Higher threshold
    nms_thresh=0.3,   # Stricter NMS
    input_size=(1024, 1024)  # Higher resolution
)

faces = detector.detect(image)
```

### Batch Processing

```python
from uniface import create_detector
import cv2
import os

detector = create_detector('retinaface')

# Process multiple images
image_dir = 'images/'
for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.png')):
        image = cv2.imread(os.path.join(image_dir, filename))
        faces = detector.detect(image)
        print(f"{filename}: {len(faces)} faces")
```

---

## Face Recognition

### Extract Embeddings

```python
from uniface import create_detector, create_recognizer
import cv2

# Initialize
detector = create_detector('retinaface')
recognizer = create_recognizer('arcface')

# Load image
image = cv2.imread('person.jpg')

# Detect faces
faces = detector.detect(image)

if faces:
    # Get embedding for first face
    landmarks = faces[0]['landmarks']
    embedding = recognizer.get_normalized_embedding(image, landmarks)
    print(f"Embedding shape: {embedding.shape}")  # (512,)
```

### Different Recognition Models

```python
from uniface import create_recognizer
from uniface.constants import ArcFaceWeights, MobileFaceWeights

# ArcFace - State-of-the-art accuracy
recognizer = create_recognizer('arcface', model_name=ArcFaceWeights.RESNET)

# MobileFace - Lightweight for edge devices
recognizer = create_recognizer('mobileface', model_name=MobileFaceWeights.MNET_V3_SMALL)

# SphereFace - Alternative approach
recognizer = create_recognizer('sphereface')
```

---

## Face Comparison

### Compare Two Faces

```python
from uniface import create_detector, create_recognizer, compute_similarity
import cv2

detector = create_detector('retinaface')
recognizer = create_recognizer('arcface')

# Load two images
image1 = cv2.imread('person1.jpg')
image2 = cv2.imread('person2.jpg')

# Detect and get embeddings
faces1 = detector.detect(image1)
faces2 = detector.detect(image2)

if faces1 and faces2:
    emb1 = recognizer.get_normalized_embedding(image1, faces1[0]['landmarks'])
    emb2 = recognizer.get_normalized_embedding(image2, faces2[0]['landmarks'])

    # Compute similarity
    similarity = compute_similarity(emb1, emb2)
    print(f"Similarity: {similarity:.4f}")

    # Typical threshold: 0.5 for same person
    if similarity > 0.5:
        print("Same person!")
    else:
        print("Different people")
```

### Build a Face Database

```python
from uniface import create_detector, create_recognizer, compute_similarity
import cv2
import numpy as np
import os

detector = create_detector('retinaface')
recognizer = create_recognizer('arcface')

# Build database from folder structure:
# faces_db/
#   person1/
#     photo1.jpg
#     photo2.jpg
#   person2/
#     photo1.jpg

database = {}  # name -> list of embeddings

db_path = 'faces_db/'
for person_name in os.listdir(db_path):
    person_path = os.path.join(db_path, person_name)
    if os.path.isdir(person_path):
        embeddings = []
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            image = cv2.imread(img_path)
            faces = detector.detect(image)
            if faces:
                emb = recognizer.get_normalized_embedding(image, faces[0]['landmarks'])
                embeddings.append(emb)
        if embeddings:
            # Store average embedding
            database[person_name] = np.mean(embeddings, axis=0)

print(f"Database contains {len(database)} people")

# Search for a face
query_image = cv2.imread('query.jpg')
query_faces = detector.detect(query_image)

if query_faces:
    query_emb = recognizer.get_normalized_embedding(query_image, query_faces[0]['landmarks'])

    best_match = None
    best_score = 0

    for name, db_emb in database.items():
        score = compute_similarity(query_emb, db_emb)
        if score > best_score:
            best_score = score
            best_match = name

    if best_score > 0.5:
        print(f"Match found: {best_match} (score: {best_score:.4f})")
    else:
        print("No match found")
```

---

## Age & Gender Prediction

### Basic Usage

```python
from uniface import create_detector, AgeGender
import cv2

detector = create_detector('retinaface')
age_gender = AgeGender()

image = cv2.imread('photo.jpg')
faces = detector.detect(image)

for face in faces:
    bbox = face['bbox']
    gender, age = age_gender.predict(image, bbox)

    gender_str = 'Female' if gender == 0 else 'Male'
    print(f"Age: {age}, Gender: {gender_str}")
```

### Combined with Detection

```python
from uniface import FaceAnalyzer, create_detector, AgeGender
import cv2

# Setup
detector = create_detector('retinaface')
age_gender = AgeGender()
analyzer = FaceAnalyzer(detector, age_gender=age_gender)

# Analyze
image = cv2.imread('group_photo.jpg')
faces = analyzer.analyze(image)

for i, face in enumerate(faces):
    print(f"Face {i+1}: Age={face.age}, Gender={face.sex}")
```

---

## Complete Analysis Pipeline

### Full FaceAnalyzer Example

```python
from uniface import (
    FaceAnalyzer,
    create_detector,
    create_recognizer,
    AgeGender,
    draw_detections,
    enable_logging
)
import cv2
import logging

# Enable logging for debugging
enable_logging(level=logging.INFO)

# Create components
detector = create_detector('retinaface', conf_thresh=0.7)
recognizer = create_recognizer('arcface')
age_gender = AgeGender()

# Create unified analyzer
analyzer = FaceAnalyzer(
    detector=detector,
    recognizer=recognizer,
    age_gender=age_gender
)

# Process image
image = cv2.imread('team_photo.jpg')
faces = analyzer.analyze(image)

# Display results
print(f"\nFound {len(faces)} faces:\n")
for i, face in enumerate(faces):
    print(f"Face {i+1}:")
    print(f"  - Confidence: {face.confidence:.3f}")
    print(f"  - Position: {face.bbox_xywh[:2].astype(int)}")
    print(f"  - Size: {face.bbox_xywh[2:4].astype(int)}")
    print(f"  - Age: {face.age}")
    print(f"  - Gender: {face.sex}")
    print(f"  - Embedding: {'Yes' if face.embedding is not None else 'No'}")
    print()

# Compare faces in the image
if len(faces) >= 2:
    print("Face similarities:")
    for i in range(len(faces)):
        for j in range(i+1, len(faces)):
            sim = faces[i].compute_similarity(faces[j])
            print(f"  Face {i+1} vs Face {j+1}: {sim:.4f}")

# Visualize
result = draw_detections(image, [f.to_dict() for f in faces])
cv2.imwrite('analyzed_result.jpg', result)
```

---

## Working with Landmarks

### 5-Point Landmarks (from Detection)

```python
from uniface import create_detector
import cv2

detector = create_detector('retinaface')
image = cv2.imread('face.jpg')
faces = detector.detect(image)

if faces:
    landmarks = faces[0]['landmarks']  # Shape: (5, 2)

    # Landmark indices:
    # 0: Left eye
    # 1: Right eye
    # 2: Nose tip
    # 3: Left mouth corner
    # 4: Right mouth corner

    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]

    print(f"Left eye: {left_eye}")
    print(f"Right eye: {right_eye}")
    print(f"Nose: {nose}")
```

### 106-Point Landmarks

```python
from uniface import create_detector, create_landmarker
import cv2

detector = create_detector('retinaface')
landmarker = create_landmarker()

image = cv2.imread('face.jpg')
faces = detector.detect(image)

if faces:
    bbox = faces[0]['bbox']
    landmarks_106 = landmarker.get_landmarks(image, bbox)
    print(f"106-point landmarks shape: {landmarks_106.shape}")  # (106, 2)
```

### Face Alignment

```python
from uniface import create_detector, face_alignment
import cv2

detector = create_detector('retinaface')
image = cv2.imread('tilted_face.jpg')
faces = detector.detect(image)

if faces:
    landmarks = faces[0]['landmarks']

    # Align face to 112x112 (standard for recognition)
    aligned = face_alignment(image, landmarks, output_size=(112, 112))
    cv2.imwrite('aligned_face.jpg', aligned)
```

---

## Visualization

### Draw Detection Results

```python
from uniface import detect_faces, draw_detections
import cv2

image = cv2.imread('photo.jpg')
faces = detect_faces(image)

# Draw with landmarks
result = draw_detections(image, faces, draw_landmarks=True)
cv2.imwrite('with_landmarks.jpg', result)

# Draw without landmarks
result = draw_detections(image, faces, draw_landmarks=False)
cv2.imwrite('boxes_only.jpg', result)
```

### Custom Visualization

```python
import cv2
import numpy as np
from uniface import detect_faces

def draw_custom(image, faces):
    result = image.copy()

    for face in faces:
        bbox = face['bbox'].astype(int)
        landmarks = face['landmarks']
        confidence = face['confidence']

        # Draw box with color based on confidence
        color = (0, int(255 * confidence), int(255 * (1 - confidence)))
        cv2.rectangle(result, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # Draw confidence text
        text = f"{confidence:.2f}"
        cv2.putText(result, text, (bbox[0], bbox[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw landmarks with different colors
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(result, (int(x), int(y)), 3, colors[i], -1)

    return result

image = cv2.imread('photo.jpg')
faces = detect_faces(image)
result = draw_custom(image, faces)
cv2.imwrite('custom_viz.jpg', result)
```

---

## Performance Optimization

### Backend Selection

```python
from uniface.backend import Backend, set_backend, get_available_backends

# Check available backends
print(get_available_backends())

# Use MLX on Apple Silicon (fastest)
set_backend(Backend.MLX)

# Use ONNX for compatibility
set_backend(Backend.ONNX)

# Auto-select best available
set_backend(Backend.AUTO)
```

### Model Selection for Speed

```python
from uniface import create_detector, create_recognizer
from uniface.constants import RetinaFaceWeights, MobileFaceWeights

# Fastest detector (smallest model)
fast_detector = create_detector(
    'retinaface',
    model_name=RetinaFaceWeights.MNET_025
)

# Fastest recognizer
fast_recognizer = create_recognizer(
    'mobileface',
    model_name=MobileFaceWeights.MNET_025
)
```

### Input Size Optimization

```python
from uniface import create_detector

# Lower resolution = faster processing
detector = create_detector(
    'retinaface',
    input_size=(320, 320)  # Default is (640, 640)
)
```

### Detector Caching

```python
from uniface import detect_faces

# detect_faces automatically caches detector instances
# Same configuration = reuses existing detector
for image in images:
    faces = detect_faces(image, method='retinaface', conf_thresh=0.5)
```

---

## Troubleshooting

### Common Issues

#### 1. No faces detected

```python
# Try lowering confidence threshold
faces = detect_faces(image, conf_thresh=0.3)

# Or try a different detector
faces = detect_faces(image, method='scrfd')
```

#### 2. MLX not available

```python
from uniface.backend import is_backend_available, Backend

if not is_backend_available(Backend.MLX):
    print("MLX requires Apple Silicon (M1/M2/M3/M4)")
    print("Install MLX: pip install mlx")
```

#### 3. Model download issues

```python
from uniface import verify_model_weights
from uniface.constants import RetinaFaceWeights

# Force re-download
import os
model_path = os.path.expanduser('~/.uniface/models/retinaface_mv2.onnx')
if os.path.exists(model_path):
    os.remove(model_path)

# Re-download with verification
verify_model_weights(RetinaFaceWeights.MNET_V2)
```

#### 4. Memory issues

```python
# Use smaller models
from uniface import create_detector, create_recognizer
from uniface.constants import RetinaFaceWeights, MobileFaceWeights

detector = create_detector('retinaface', model_name=RetinaFaceWeights.MNET_025)
recognizer = create_recognizer('mobileface', model_name=MobileFaceWeights.MNET_025)

# Process images sequentially instead of loading all at once
import cv2
import gc

for img_path in image_paths:
    image = cv2.imread(img_path)
    faces = detector.detect(image)
    # Process...
    del image
    gc.collect()
```

### Debug Mode

```python
from uniface import enable_logging
import logging

# Enable verbose logging
enable_logging(level=logging.DEBUG)

# Now all operations will print debug info
from uniface import detect_faces
faces = detect_faces(image)
```

---

## Best Practices

### 1. Choose the Right Model

| Use Case | Detector | Recognizer |
|----------|----------|------------|
| Production accuracy | RetinaFace RESNET34 | ArcFace RESNET |
| Real-time processing | SCRFD 500M | MobileFace MNET_V3_SMALL |
| Mobile/Edge devices | RetinaFace MNET_025 | MobileFace MNET_025 |
| Balanced | RetinaFace MNET_V2 | ArcFace MNET |

### 2. Use Appropriate Thresholds

```python
# High precision (fewer false positives)
detector = create_detector('retinaface', conf_thresh=0.8, nms_thresh=0.3)

# High recall (fewer missed faces)
detector = create_detector('retinaface', conf_thresh=0.3, nms_thresh=0.5)
```

### 3. Handle Edge Cases

```python
def safe_analyze(image, analyzer):
    """Safely analyze image with error handling."""
    if image is None:
        return []

    if len(image.shape) != 3:
        return []

    try:
        return analyzer.analyze(image)
    except Exception as e:
        print(f"Analysis failed: {e}")
        return []
```

### 4. Normalize Similarity Scores

```python
def is_same_person(embedding1, embedding2, threshold=0.5):
    """Check if two embeddings belong to the same person."""
    from uniface import compute_similarity
    similarity = compute_similarity(embedding1, embedding2)
    return similarity > threshold, similarity
```

---

*User Guide for MLX-UniFace v1.3.1*
