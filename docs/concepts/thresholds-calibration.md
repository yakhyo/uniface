# Thresholds & Calibration

This page explains how to tune detection and recognition thresholds for your use case.

---

## Detection Thresholds

### Confidence Threshold

Controls minimum confidence for face detection:

```python
from uniface.detection import RetinaFace

# Default (balanced)
detector = RetinaFace(confidence_threshold=0.5)

# High precision (fewer false positives)
detector = RetinaFace(confidence_threshold=0.8)

# High recall (catch more faces)
detector = RetinaFace(confidence_threshold=0.3)
```

**Guidelines:**

| Threshold | Use Case |
|-----------|----------|
| 0.3 - 0.4 | Maximum recall (research, analysis) |
| 0.5 - 0.6 | Balanced (default, general use) |
| 0.7 - 0.9 | High precision (production, security) |

---

### NMS Threshold

Non-Maximum Suppression removes overlapping detections:

```python
# Default
detector = RetinaFace(nms_threshold=0.4)

# Stricter (fewer overlapping boxes)
detector = RetinaFace(nms_threshold=0.3)

# Looser (for crowded scenes)
detector = RetinaFace(nms_threshold=0.5)
```

---

### Input Size

Affects detection accuracy and speed:

```python
# Faster, lower accuracy
detector = RetinaFace(input_size=(320, 320))

# Balanced (default)
detector = RetinaFace(input_size=(640, 640))

# Higher accuracy, slower
detector = RetinaFace(input_size=(1280, 1280))
```

!!! tip "Dynamic Size"
    For RetinaFace, enable dynamic input for variable image sizes:
    ```python
    detector = RetinaFace(dynamic_size=True)
    ```

---

## Recognition Thresholds

### Similarity Threshold

For identity verification (same person check):

```python
import numpy as np
from uniface.face_utils import compute_similarity

similarity = compute_similarity(embedding1, embedding2)

# Threshold interpretation
if similarity > 0.6:
    print("Same person (high confidence)")
elif similarity > 0.4:
    print("Uncertain (manual review)")
else:
    print("Different people")
```

**Recommended thresholds:**

| Threshold | Decision | False Accept Rate |
|-----------|----------|-------------------|
| 0.4 | Low security | Higher FAR |
| 0.5 | Balanced | Moderate FAR |
| 0.6 | High security | Lower FAR |
| 0.7 | Very strict | Very low FAR |

---

### Calibration for Your Dataset

Test on your data to find optimal thresholds:

```python
import numpy as np

def calibrate_threshold(same_pairs, diff_pairs, recognizer, detector):
    """Find optimal threshold for your dataset."""
    same_scores = []
    diff_scores = []

    # Compute similarities for same-person pairs
    for img1_path, img2_path in same_pairs:
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        faces1 = detector.detect(img1)
        faces2 = detector.detect(img2)

        if faces1 and faces2:
            emb1 = recognizer.get_normalized_embedding(img1, faces1[0].landmarks)
            emb2 = recognizer.get_normalized_embedding(img2, faces2[0].landmarks)
            same_scores.append(np.dot(emb1, emb2.T)[0][0])

    # Compute similarities for different-person pairs
    for img1_path, img2_path in diff_pairs:
        # ... similar process
        diff_scores.append(similarity)

    # Find optimal threshold
    thresholds = np.arange(0.3, 0.8, 0.05)
    best_threshold = 0.5
    best_accuracy = 0

    for thresh in thresholds:
        tp = sum(1 for s in same_scores if s >= thresh)
        tn = sum(1 for s in diff_scores if s < thresh)
        accuracy = (tp + tn) / (len(same_scores) + len(diff_scores))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = thresh

    return best_threshold, best_accuracy
```

---

## Anti-Spoofing Thresholds

The MiniFASNet model returns a confidence score:

```python
from uniface.spoofing import MiniFASNet

spoofer = MiniFASNet()
result = spoofer.predict(image, face.bbox)

# Default threshold (0.5)
if result.is_real:  # confidence > 0.5
    print("Real face")

# Custom threshold for high security
SPOOF_THRESHOLD = 0.7
if result.confidence > SPOOF_THRESHOLD:
    print("Real face (high confidence)")
else:
    print("Potentially fake")
```

---

## Attribute Model Confidence

### Emotion

```python
result = emotion_predictor.predict(image, landmarks)

# Filter low-confidence predictions
if result.confidence > 0.6:
    print(f"Emotion: {result.emotion}")
else:
    print("Uncertain emotion")
```

---

## Visualization Threshold

For drawing detections, filter by confidence:

```python
from uniface.draw import draw_detections

# Only draw high-confidence detections
bboxes = [f.bbox for f in faces if f.confidence > 0.7]
scores = [f.confidence for f in faces if f.confidence > 0.7]
landmarks = [f.landmarks for f in faces if f.confidence > 0.7]

draw_detections(
    image=image,
    bboxes=bboxes,
    scores=scores,
    landmarks=landmarks,
    vis_threshold=0.6  # Additional visualization filter
)
```

---

## Summary

| Parameter | Default | Range | Lower = | Higher = |
|-----------|---------|-------|---------|----------|
| `confidence_threshold` | 0.5 | 0.1-0.9 | More detections | Fewer false positives |
| `nms_threshold` | 0.4 | 0.1-0.7 | Fewer overlaps | More overlapping boxes |
| Similarity threshold | 0.6 | 0.3-0.8 | More matches (FAR↑) | Fewer matches (FRR↑) |
| Spoof confidence | 0.5 | 0.3-0.9 | More "real" | Stricter liveness |

---

## Next Steps

- [Detection Module](../modules/detection.md) - Detection model options
- [Recognition Module](../modules/recognition.md) - Recognition model options
