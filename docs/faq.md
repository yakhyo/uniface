# FAQ

Frequently asked questions.

---

## General

### What is UniFace?

A Python library for face analysis: detection, recognition, landmarks, attributes, parsing, gaze estimation, anti-spoofing, and privacy protection.

### What are the requirements?

- Python 3.11+
- Works on macOS, Linux, Windows

### Is GPU required?

No. CPU works fine. GPU (CUDA) provides faster inference.

---

## Models

### Where are models stored?

```
~/.uniface/models/
```

### How to use offline?

Pre-download models:

```python
from uniface.model_store import verify_model_weights
from uniface.constants import RetinaFaceWeights

verify_model_weights(RetinaFaceWeights.MNET_V2)
```

### Which detection model is best?

| Use Case | Model |
|----------|-------|
| Balanced | RetinaFace MNET_V2 |
| Accuracy | SCRFD 10G |
| Speed | YOLOv5n-Face |

---

## Usage

### What image format?

BGR (OpenCV default):

```python
image = cv2.imread("photo.jpg")  # BGR
```

### How to compare faces?

```python
from uniface import compute_similarity

similarity = compute_similarity(emb1, emb2)
if similarity > 0.6:
    print("Same person")
```

### How to get age and gender?

```python
from uniface import AgeGender

predictor = AgeGender()
result = predictor.predict(image, face.bbox)
print(f"{result.sex}, {result.age}")
```

---

## Performance

### How to speed up detection?

1. Use smaller input:
   ```python
   detector = RetinaFace(input_size=(320, 320))
   ```

2. Skip frames in video:
   ```python
   if frame_count % 3 == 0:
       faces = detector.detect(frame)
   ```

3. Use GPU:
   ```bash
   pip install uniface[gpu]
   ```

---

## Accuracy

### Detection threshold?

Default: 0.5

- Higher (0.7+): Fewer false positives
- Lower (0.3): More detections

### Similarity threshold?

| Threshold | Meaning |
|-----------|---------|
| > 0.6 | Same person |
| 0.4-0.6 | Uncertain |
| < 0.4 | Different |

---

## Privacy

### How to blur faces?

```python
from uniface.privacy import anonymize_faces

result = anonymize_faces(image, method='pixelate')
```

### Available blur methods?

`pixelate`, `gaussian`, `blackout`, `elliptical`, `median`
