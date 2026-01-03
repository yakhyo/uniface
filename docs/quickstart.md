# Quickstart

Get up and running with UniFace in 5 minutes. This guide covers the most common use cases.

---

## Face Detection

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

## Visualize Detections

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
```

---

## Face Recognition

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
```

!!! tip "Similarity Thresholds"
    - `> 0.6`: Same person (high confidence)
    - `0.4 - 0.6`: Uncertain (manual review)
    - `< 0.4`: Different people

---

## Age & Gender Detection

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
```

**Output:**

```
Face 1: Male, 32 years old
Face 2: Female, 28 years old
```

---

## FairFace Attributes

Detect race, gender, and age group:

```python
import cv2
from uniface import RetinaFace, FairFace

detector = RetinaFace()
fairface = FairFace()

image = cv2.imread("photo.jpg")
faces = detector.detect(image)

for i, face in enumerate(faces):
    result = fairface.predict(image, face.bbox)
    print(f"Face {i+1}: {result.sex}, {result.age_group}, {result.race}")
```

**Output:**

```
Face 1: Male, 30-39, East Asian
Face 2: Female, 20-29, White
```

---

## Facial Landmarks (106 Points)

```python
import cv2
from uniface import RetinaFace, Landmark106

detector = RetinaFace()
landmarker = Landmark106()

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

## Gaze Estimation

```python
import cv2
import numpy as np
from uniface import RetinaFace, MobileGaze
from uniface.visualization import draw_gaze

detector = RetinaFace()
gaze_estimator = MobileGaze()

image = cv2.imread("photo.jpg")
faces = detector.detect(image)

for i, face in enumerate(faces):
    x1, y1, x2, y2 = map(int, face.bbox[:4])
    face_crop = image[y1:y2, x1:x2]

    if face_crop.size > 0:
        result = gaze_estimator.estimate(face_crop)
        print(f"Face {i+1}: pitch={np.degrees(result.pitch):.1f}°, yaw={np.degrees(result.yaw):.1f}°")

        # Draw gaze direction
        draw_gaze(image, face.bbox, result.pitch, result.yaw)

cv2.imwrite("gaze_output.jpg", image)
```

---

## Face Parsing

Segment face into semantic components:

```python
import cv2
import numpy as np
from uniface.parsing import BiSeNet
from uniface.visualization import vis_parsing_maps

parser = BiSeNet()

# Load face image (already cropped)
face_image = cv2.imread("face.jpg")

# Parse face into 19 components
mask = parser.parse(face_image)

# Visualize with overlay
face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
vis_result = vis_parsing_maps(face_rgb, mask, save_image=False)

print(f"Detected {len(np.unique(mask))} facial components")
```

---

## Face Anonymization

Blur faces for privacy protection:

```python
from uniface.privacy import anonymize_faces
import cv2

# One-liner: automatic detection and blurring
image = cv2.imread("group_photo.jpg")
anonymized = anonymize_faces(image, method='pixelate')
cv2.imwrite("anonymized.jpg", anonymized)
```

**Manual control:**

```python
from uniface import RetinaFace
from uniface.privacy import BlurFace

detector = RetinaFace()
blurrer = BlurFace(method='gaussian', blur_strength=5.0)

faces = detector.detect(image)
anonymized = blurrer.anonymize(image, faces)
```

**Available methods:**

| Method | Description |
|--------|-------------|
| `pixelate` | Blocky effect (news media standard) |
| `gaussian` | Smooth, natural blur |
| `blackout` | Solid color boxes (maximum privacy) |
| `elliptical` | Soft oval blur (natural face shape) |
| `median` | Edge-preserving blur |

---

## Face Anti-Spoofing

Detect real vs. fake faces:

```python
import cv2
from uniface import RetinaFace
from uniface.spoofing import MiniFASNet

detector = RetinaFace()
spoofer = MiniFASNet()

image = cv2.imread("photo.jpg")
faces = detector.detect(image)

for i, face in enumerate(faces):
    result = spoofer.predict(image, face.bbox)
    label = 'Real' if result.is_real else 'Fake'
    print(f"Face {i+1}: {label} ({result.confidence:.1%})")
```

---

## Webcam Demo

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

    faces = detector.detect(frame)

    bboxes = [f.bbox for f in faces]
    scores = [f.confidence for f in faces]
    landmarks = [f.landmarks for f in faces]
    draw_detections(image=frame, bboxes=bboxes, scores=scores, landmarks=landmarks)

    cv2.imshow("UniFace - Press 'q' to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Model Selection

For detailed model comparisons and benchmarks, see the [Model Zoo](models.md).

**Available models by task:**

| Task | Available Models |
|------|------------------|
| Detection | `RetinaFace`, `SCRFD`, `YOLOv5Face`, `YOLOv8Face` |
| Recognition | `ArcFace`, `AdaFace`, `MobileFace`, `SphereFace` |
| Gaze | `MobileGaze` (ResNet18/34/50, MobileNetV2, MobileOneS0) |
| Parsing | `BiSeNet` (ResNet18/34) |
| Attributes | `AgeGender`, `FairFace`, `Emotion` |
| Anti-Spoofing | `MiniFASNet` (V1SE, V2) |

---

## Common Issues

### Models Not Downloading

```python
from uniface.model_store import verify_model_weights
from uniface.constants import RetinaFaceWeights

# Manually download a model
model_path = verify_model_weights(RetinaFaceWeights.MNET_V2)
print(f"Model downloaded to: {model_path}")
```

### Check Hardware Acceleration

```python
import onnxruntime as ort
print("Available providers:", ort.get_available_providers())

# macOS M-series should show: ['CoreMLExecutionProvider', ...]
# NVIDIA GPU should show: ['CUDAExecutionProvider', ...]
```

### Slow Performance on Mac

Verify you're using the ARM64 build of Python:

```bash
python -c "import platform; print(platform.machine())"
# Should show: arm64 (not x86_64)
```

### Import Errors

```python
# Correct imports
from uniface.detection import RetinaFace
from uniface.recognition import ArcFace
from uniface.landmark import Landmark106

# Also works (re-exported at package level)
from uniface import RetinaFace, ArcFace, Landmark106
```

---

## Next Steps

- [Model Zoo](models.md) - All models, benchmarks, and selection guide
- [API Reference](modules/detection.md) - Explore individual modules and their APIs
- [Tutorials](recipes/image-pipeline.md) - Step-by-step examples for common workflows
- [Guides](concepts/overview.md) - Learn about the architecture and design principles
