# Coordinate Systems

This page explains the coordinate formats used in UniFace.

---

## Image Coordinates

All coordinates use **pixel-based, top-left origin**:

```
(0, 0) ────────────────► x (width)
   │
   │    Image
   │
   ▼
   y (height)
```

---

## Bounding Box Format

Bounding boxes use `[x1, y1, x2, y2]` format (top-left and bottom-right corners):

```
(x1, y1) ─────────────────┐
    │                     │
    │      Face           │
    │                     │
    └─────────────────────┘ (x2, y2)
```

### Accessing Coordinates

```python
face = faces[0]

# Direct access
x1, y1, x2, y2 = face.bbox

# As properties
bbox_xyxy = face.bbox_xyxy  # [x1, y1, x2, y2]
bbox_xywh = face.bbox_xywh  # [x1, y1, width, height]
```

### Conversion

```python
import numpy as np

# xyxy → xywh
def xyxy_to_xywh(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([x1, y1, x2 - x1, y2 - y1])

# xywh → xyxy
def xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h])
```

---

## Landmarks

### 5-Point Landmarks (Detection)

Returned by all detection models:

```python
landmarks = face.landmarks  # Shape: (5, 2)
```

| Index | Point |
|-------|-------|
| 0 | Left Eye |
| 1 | Right Eye |
| 2 | Nose Tip |
| 3 | Left Mouth Corner |
| 4 | Right Mouth Corner |

```
      0 ●           ● 1

            ● 2

        3 ●     ● 4
```

### 106-Point Landmarks

Returned by `Landmark106`:

```python
from uniface import Landmark106

landmarker = Landmark106()
landmarks = landmarker.get_landmarks(image, face.bbox)
# Shape: (106, 2)
```

**Landmark Groups:**

| Range | Group | Points |
|-------|-------|--------|
| 0-32 | Face Contour | 33 |
| 33-50 | Eyebrows | 18 |
| 51-62 | Nose | 12 |
| 63-86 | Eyes | 24 |
| 87-105 | Mouth | 19 |

---

## Face Crop

To crop a face from an image:

```python
def crop_face(image, bbox, margin=0):
    """Crop face with optional margin."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    # Add margin
    if margin > 0:
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, x1 - int(bw * margin))
        y1 = max(0, y1 - int(bh * margin))
        x2 = min(w, x2 + int(bw * margin))
        y2 = min(h, y2 + int(bh * margin))

    return image[y1:y2, x1:x2]

# Usage
face_crop = crop_face(image, face.bbox, margin=0.1)
```

---

## Gaze Angles

Gaze estimation returns pitch and yaw in **radians**:

```python
result = gaze_estimator.estimate(face_crop)

# Angles in radians
pitch = result.pitch  # Vertical: + = up, - = down
yaw = result.yaw      # Horizontal: + = right, - = left

# Convert to degrees
import numpy as np
pitch_deg = np.degrees(pitch)
yaw_deg = np.degrees(yaw)
```

**Angle Reference:**

```
          pitch = +90° (up)
               │
               │
yaw = -90° ────┼──── yaw = +90°
(left)         │      (right)
               │
          pitch = -90° (down)
```

---

## Face Alignment

Face alignment uses 5-point landmarks to normalize face orientation:

```python
from uniface import face_alignment

# Align face to standard template
aligned_face = face_alignment(image, face.landmarks)
# Output: 112x112 aligned face image
```

The alignment transforms faces to a canonical pose for better recognition accuracy.

---

## Next Steps

- [Inputs & Outputs](inputs-outputs.md) - Data types reference
- [Recognition Module](../modules/recognition.md) - Face recognition details
