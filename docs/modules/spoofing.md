# Anti-Spoofing

Face anti-spoofing detects whether a face is real (live) or fake (photo, video replay, mask).

---

## Available Models

| Model | Size | Notes |
|-------|------|-------|
| MiniFASNet V1SE | 1.2 MB | Squeeze-and-Excitation variant |
| **MiniFASNet V2** ⭐ | 1.2 MB | Improved version (recommended) |

---

## Basic Usage

```python
import cv2
from uniface import RetinaFace
from uniface.spoofing import MiniFASNet

detector = RetinaFace()
spoofer = MiniFASNet()

image = cv2.imread("photo.jpg")
faces = detector.detect(image)

for face in faces:
    result = spoofer.predict(image, face.bbox)

    label = "Real" if result.is_real else "Fake"
    print(f"{label}: {result.confidence:.1%}")
```

---

## Output Format

```python
result = spoofer.predict(image, face.bbox)

# SpoofingResult dataclass
result.is_real     # True = real, False = fake
result.confidence  # 0.0 to 1.0
```

---

## Model Variants

```python
from uniface.spoofing import MiniFASNet
from uniface.constants import MiniFASNetWeights

# Default (V2, recommended)
spoofer = MiniFASNet()

# V1SE variant
spoofer = MiniFASNet(model_name=MiniFASNetWeights.V1SE)
```

| Variant | Size | Scale Factor |
|---------|------|--------------|
| V1SE | 1.2 MB | 4.0 |
| **V2** ⭐ | 1.2 MB | 2.7 |

---

## Confidence Thresholds

The default threshold is 0.5. Adjust for your use case:

```python
result = spoofer.predict(image, face.bbox)

# High security (fewer false accepts)
HIGH_THRESHOLD = 0.7
if result.confidence > HIGH_THRESHOLD:
    print("Real (high confidence)")
else:
    print("Suspicious")

# Balanced
if result.is_real:  # Uses default 0.5 threshold
    print("Real")
else:
    print("Fake")
```

---

## Visualization

```python
import cv2

def draw_spoofing_result(image, face, result):
    """Draw spoofing result on image."""
    x1, y1, x2, y2 = map(int, face.bbox)

    # Color based on result
    color = (0, 255, 0) if result.is_real else (0, 0, 255)
    label = "Real" if result.is_real else "Fake"

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Draw label
    text = f"{label}: {result.confidence:.1%}"
    cv2.putText(image, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image

# Usage
for face in faces:
    result = spoofer.predict(image, face.bbox)
    image = draw_spoofing_result(image, face, result)

cv2.imwrite("spoofing_result.jpg", image)
```

---

## Real-Time Liveness Detection

```python
import cv2
from uniface import RetinaFace
from uniface.spoofing import MiniFASNet

detector = RetinaFace()
spoofer = MiniFASNet()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector.detect(frame)

    for face in faces:
        result = spoofer.predict(frame, face.bbox)

        # Draw result
        x1, y1, x2, y2 = map(int, face.bbox)
        color = (0, 255, 0) if result.is_real else (0, 0, 255)
        label = f"{'Real' if result.is_real else 'Fake'}: {result.confidence:.0%}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Liveness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Use Cases

### Access Control

```python
def verify_liveness(image, face, spoofer, threshold=0.6):
    """Verify face is real for access control."""
    result = spoofer.predict(image, face.bbox)

    if result.is_real and result.confidence > threshold:
        return True, result.confidence
    return False, result.confidence

# Usage
is_live, confidence = verify_liveness(image, face, spoofer)
if is_live:
    print(f"Access granted (confidence: {confidence:.1%})")
else:
    print(f"Access denied - possible spoof attempt")
```

### Multi-Frame Verification

For higher security, verify across multiple frames:

```python
def verify_liveness_multiframe(frames, detector, spoofer, min_real=3):
    """Verify liveness across multiple frames."""
    real_count = 0

    for frame in frames:
        faces = detector.detect(frame)
        if not faces:
            continue

        result = spoofer.predict(frame, faces[0].bbox)
        if result.is_real:
            real_count += 1

    return real_count >= min_real

# Collect frames and verify
frames = []
for _ in range(5):
    ret, frame = cap.read()
    if ret:
        frames.append(frame)

is_verified = verify_liveness_multiframe(frames, detector, spoofer)
```

---

## Attack Types Detected

MiniFASNet can detect various spoof attacks:

| Attack Type | Detection |
|-------------|-----------|
| Printed photos | ✅ |
| Screen replay | ✅ |
| Video replay | ✅ |
| Paper masks | ✅ |
| 3D masks | Limited |

!!! warning "Limitations"
    - High-quality 3D masks may not be detected
    - Performance varies with lighting and image quality
    - Always combine with other verification methods for high-security applications

---

## Command-Line Tool

```bash
# Image
python tools/spoofing.py --source photo.jpg

# Webcam
python tools/spoofing.py --source 0
```

---

## Factory Function

```python
from uniface import create_spoofer

spoofer = create_spoofer()  # Returns MiniFASNet
```

---

## Next Steps

- [Privacy](privacy.md) - Face anonymization
- [Detection](detection.md) - Face detection
- [Recognition](recognition.md) - Face recognition
