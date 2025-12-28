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
    print(f"  Confidence: {face.confidence:.2f}")
    print(f"  BBox: {face.bbox}")
    print(f"  Landmarks: {len(face.landmarks)} points")
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
bboxes = [f.bbox for f in faces]
scores = [f.confidence for f in faces]
landmarks = [f.landmarks for f in faces]

# Draw on image
draw_detections(
    image=image,
    bboxes=bboxes,
    scores=scores,
    landmarks=landmarks,
    vis_threshold=0.6,
)

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
    emb1 = recognizer.get_normalized_embedding(image1, faces1[0].landmarks)
    emb2 = recognizer.get_normalized_embedding(image2, faces2[0].landmarks)

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
    bboxes = [f.bbox for f in faces]
    scores = [f.confidence for f in faces]
    landmarks = [f.landmarks for f in faces]
    draw_detections(
        image=frame,
        bboxes=bboxes,
        scores=scores,
        landmarks=landmarks,
    )

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
    result = age_gender.predict(image, face.bbox)
    print(f"Face {i+1}: {result.sex}, {result.age} years old")
    # result.gender: 0=Female, 1=Male
    # result.sex: "Female" or "Male"
    # result.age: age in years
```

**Output:**

```
Face 1: Male, 32 years old
Face 2: Female, 28 years old
```

---

## 5b. FairFace Attributes (2 minutes)

Detect race, gender, and age group with balanced demographics:

```python
import cv2
from uniface import RetinaFace, FairFace

# Initialize models
detector = RetinaFace()
fairface = FairFace()

# Load image
image = cv2.imread("photo.jpg")
faces = detector.detect(image)

# Predict attributes
for i, face in enumerate(faces):
    result = fairface.predict(image, face.bbox)
    print(f"Face {i+1}: {result.sex}, {result.age_group}, {result.race}")
    # result.gender: 0=Female, 1=Male
    # result.sex: "Female" or "Male"
    # result.age_group: "20-29", "30-39", etc.
    # result.race: "East Asian", "White", etc.
```

**Output:**

```
Face 1: Male, 30-39, East Asian
Face 2: Female, 20-29, White
```

**Race Categories:** White, Black, Latino Hispanic, East Asian, Southeast Asian, Indian, Middle Eastern

**Age Groups:** 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+

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
    landmarks = landmarker.get_landmarks(image, faces[0].bbox)
    print(f"Detected {len(landmarks)} landmarks")

    # Draw landmarks
    for x, y in landmarks.astype(int):
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    cv2.imwrite("landmarks.jpg", image)
```

---

## 7. Gaze Estimation (2 minutes)

Estimate where a person is looking:

```python
import cv2
import numpy as np
from uniface import RetinaFace, MobileGaze
from uniface.visualization import draw_gaze

# Initialize models
detector = RetinaFace()
gaze_estimator = MobileGaze()

# Load image
image = cv2.imread("photo.jpg")
faces = detector.detect(image)

# Estimate gaze for each face
for i, face in enumerate(faces):
    x1, y1, x2, y2 = map(int, face.bbox[:4])
    face_crop = image[y1:y2, x1:x2]

    if face_crop.size > 0:
        pitch, yaw = gaze_estimator.estimate(face_crop)
        print(f"Face {i+1}: pitch={np.degrees(pitch):.1f}°, yaw={np.degrees(yaw):.1f}°")

        # Draw gaze direction
        draw_gaze(image, face.bbox, pitch, yaw)

cv2.imwrite("gaze_output.jpg", image)
```

**Output:**

```
Face 1: pitch=5.2°, yaw=-12.3°
Face 2: pitch=-8.1°, yaw=15.7°
```

---

## 8. Face Parsing (2 minutes)

Segment face into semantic components (skin, eyes, nose, mouth, hair, etc.):

```python
import cv2
import numpy as np
from uniface.parsing import BiSeNet
from uniface.visualization import vis_parsing_maps

# Initialize parser
parser = BiSeNet()  # Uses ResNet18 by default

# Load face image (already cropped)
face_image = cv2.imread("face.jpg")

# Parse face into 19 components
mask = parser.parse(face_image)

# Visualize with overlay
face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
vis_result = vis_parsing_maps(face_rgb, mask, save_image=False)

# Convert back to BGR for saving
vis_bgr = cv2.cvtColor(vis_result, cv2.COLOR_RGB2BGR)
cv2.imwrite("parsed_face.jpg", vis_bgr)

print(f"Detected {len(np.unique(mask))} facial components")
```

**Output:**

```
Detected 12 facial components
```

**19 Facial Component Classes:**
- Background, Skin, Eyebrows (L/R), Eyes (L/R), Eye Glasses
- Ears (L/R), Ear Ring, Nose, Mouth, Lips (Upper/Lower)
- Neck, Neck Lace, Cloth, Hair, Hat

---

## 9. Face Anonymization (2 minutes)

Automatically blur faces for privacy protection:

```python
from uniface.privacy import anonymize_faces
import cv2

# One-liner: automatic detection and blurring
image = cv2.imread("group_photo.jpg")
anonymized = anonymize_faces(image, method='pixelate')
cv2.imwrite("anonymized.jpg", anonymized)
print("Faces anonymized successfully!")
```

**Manual control with custom parameters:**

```python
from uniface import RetinaFace
from uniface.privacy import BlurFace

# Initialize detector and blurrer
detector = RetinaFace()
blurrer = BlurFace(method='gaussian', blur_strength=5.0)

# Detect and anonymize
faces = detector.detect(image)
anonymized = blurrer.anonymize(image, faces)
cv2.imwrite("output.jpg", anonymized)
```

**Available blur methods:**

```python
# Pixelation (news media standard)
blurrer = BlurFace(method='pixelate', pixel_blocks=8)

# Gaussian blur (smooth, natural)
blurrer = BlurFace(method='gaussian', blur_strength=4.0)

# Black boxes (maximum privacy)
blurrer = BlurFace(method='blackout', color=(0, 0, 0))

# Elliptical blur (natural face shape)
blurrer = BlurFace(method='elliptical', blur_strength=3.0, margin=30)

# Median blur (edge-preserving)
blurrer = BlurFace(method='median', blur_strength=3.0)
```

**Webcam anonymization:**

```python
import cv2
from uniface import RetinaFace
from uniface.privacy import BlurFace

detector = RetinaFace()
blurrer = BlurFace(method='pixelate')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector.detect(frame)
    frame = blurrer.anonymize(frame, faces, inplace=True)

    cv2.imshow('Anonymized', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**Command-line tool:**

```bash
# Anonymize image with pixelation
python scripts/run_anonymization.py --image photo.jpg

# Real-time webcam anonymization
python scripts/run_anonymization.py --webcam --method gaussian

# Custom blur strength
python scripts/run_anonymization.py --image photo.jpg --method gaussian --blur-strength 5.0
```

---

## 10. Face Anti-Spoofing (2 minutes)

Detect if a face is real or fake (photo, video replay, mask):

```python
from uniface import RetinaFace
from uniface.spoofing import MiniFASNet

detector = RetinaFace()
spoofer = MiniFASNet()  # Uses V2 by default

image = cv2.imread("photo.jpg")
faces = detector.detect(image)

for i, face in enumerate(faces):
    label_idx, score = spoofer.predict(image, face.bbox)
    # label_idx: 0 = Fake, 1 = Real
    label = 'Real' if label_idx == 1 else 'Fake'
    print(f"Face {i+1}: {label} ({score:.1%})")
```

**Output:**

```
Face 1: Real (98.5%)
```

**Command-line tool:**

```bash
# Image
python scripts/run_spoofing.py --image photo.jpg

# Webcam
python scripts/run_spoofing.py --source 0
```

---

## 11. Batch Processing (3 minutes)

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

## 12. Model Selection

Choose the right model for your use case:

### Detection Models

```python
from uniface.detection import RetinaFace, SCRFD, YOLOv5Face
from uniface.constants import RetinaFaceWeights, SCRFDWeights, YOLOv5FaceWeights

# Fast detection (mobile/edge devices)
detector = RetinaFace(
    model_name=RetinaFaceWeights.MNET_025,
    confidence_threshold=0.7
)

# Balanced (recommended)
detector = RetinaFace(
    model_name=RetinaFaceWeights.MNET_V2
)

# Real-time with high accuracy
detector = YOLOv5Face(
    model_name=YOLOv5FaceWeights.YOLOV5S,
    confidence_threshold=0.6,
    nms_thresh=0.5
)

# High accuracy (server/GPU)
detector = SCRFD(
    model_name=SCRFDWeights.SCRFD_10G_KPS,
    confidence_threshold=0.5
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

### Gaze Estimation Models

```python
from uniface import MobileGaze
from uniface.constants import GazeWeights

# Default (recommended)
gaze_estimator = MobileGaze()  # Uses RESNET34

# Lightweight (mobile/edge devices)
gaze_estimator = MobileGaze(model_name=GazeWeights.MOBILEONE_S0)

# High accuracy
gaze_estimator = MobileGaze(model_name=GazeWeights.RESNET50)
```

### Face Parsing Models

```python
from uniface.parsing import BiSeNet
from uniface.constants import ParsingWeights

# Default (recommended, 50.7 MB)
parser = BiSeNet()  # Uses RESNET18

# Higher accuracy (89.2 MB)
parser = BiSeNet(model_name=ParsingWeights.RESNET34)
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

### Jupyter Notebook Examples

Explore interactive examples for common tasks:

| Example | Description | Notebook |
|---------|-------------|----------|
| **Face Detection** | Detect faces and facial landmarks | [01_face_detection.ipynb](examples/01_face_detection.ipynb) |
| **Face Alignment** | Align and crop faces for recognition | [02_face_alignment.ipynb](examples/02_face_alignment.ipynb) |
| **Face Verification** | Compare two faces to verify identity | [03_face_verification.ipynb](examples/03_face_verification.ipynb) |
| **Face Search** | Find a person in a group photo | [04_face_search.ipynb](examples/04_face_search.ipynb) |
| **Face Analyzer** | All-in-one detection, recognition & attributes | [05_face_analyzer.ipynb](examples/05_face_analyzer.ipynb) |
| **Face Parsing** | Segment face into semantic components | [06_face_parsing.ipynb](examples/06_face_parsing.ipynb) |
| **Face Anonymization** | Blur or pixelate faces for privacy protection | [07_face_anonymization.ipynb](examples/07_face_anonymization.ipynb) |
| **Gaze Estimation** | Estimate gaze direction | [08_gaze_estimation.ipynb](examples/08_gaze_estimation.ipynb) |

### Additional Resources

- **Model Benchmarks**: See [MODELS.md](MODELS.md) for performance comparisons
- **Full Documentation**: Read [README.md](README.md) for complete API reference

---

## References

- **RetinaFace Training**: [yakhyo/retinaface-pytorch](https://github.com/yakhyo/retinaface-pytorch)
- **YOLOv5-Face ONNX**: [yakhyo/yolov5-face-onnx-inference](https://github.com/yakhyo/yolov5-face-onnx-inference)
- **Face Recognition Training**: [yakhyo/face-recognition](https://github.com/yakhyo/face-recognition)
- **Gaze Estimation Training**: [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation)
- **Face Parsing Training**: [yakhyo/face-parsing](https://github.com/yakhyo/face-parsing)
- **FairFace**: [yakhyo/fairface-onnx](https://github.com/yakhyo/fairface-onnx) - Race, gender, age prediction
- **InsightFace**: [deepinsight/insightface](https://github.com/deepinsight/insightface)
