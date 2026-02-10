# Tracking

Multi-object face tracking using ByteTrack with Kalman filtering and IoU-based association.

---

## Basic Usage

```python
import cv2
import numpy as np
from uniface.detection import SCRFD
from uniface.tracking import BYTETracker, expand_bboxes
from uniface.draw import draw_tracks

detector = SCRFD()
tracker = BYTETracker(track_thresh=0.5, track_buffer=30)

cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    faces = detector.detect(frame)
    dets = np.array([[*f.bbox, f.confidence] for f in faces])
    dets = dets if len(dets) > 0 else np.empty((0, 5))

    # Expand bboxes for tracking stability
    expanded = expand_bboxes(dets, scale=1.5, image_shape=frame.shape[:2])

    # Update tracker
    tracks = tracker.update(expanded)  # (M, 5) with [x1, y1, x2, y2, track_id]

    # Assign track IDs back to Face objects
    if len(tracks) > 0 and len(faces) > 0:
        face_bboxes = np.array([f.bbox for f in faces], dtype=np.float32)
        track_cx = (tracks[:, 0] + tracks[:, 2]) / 2
        track_cy = (tracks[:, 1] + tracks[:, 3]) / 2
        face_cx = (face_bboxes[:, 0] + face_bboxes[:, 2]) / 2
        face_cy = (face_bboxes[:, 1] + face_bboxes[:, 3]) / 2

        for i in range(len(faces)):
            dists = (face_cx[i] - track_cx) ** 2 + (face_cy[i] - track_cy) ** 2
            faces[i].track_id = int(tracks[int(np.argmin(dists)), 4])

    # Draw results
    draw_tracks(image=frame, faces=[f for f in faces if f.track_id is not None])
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

Each track ID gets a deterministic color via golden-ratio hue stepping, so the same ID always has the same color across frames.

---

## BYTETracker Parameters

```python
from uniface.tracking import BYTETracker

tracker = BYTETracker(
    track_thresh=0.5,      # Detections above this are high-confidence
    track_buffer=30,       # Keep lost tracks for this many frames
    match_thresh=0.8,      # IoU threshold for first association
    low_thresh=0.1,        # Detections below this are ignored
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `track_thresh` | 0.5 | High-confidence detection threshold |
| `track_buffer` | 30 | Frames to keep lost tracks before removing |
| `match_thresh` | 0.8 | IoU threshold for matching tracks to detections |
| `low_thresh` | 0.1 | Minimum detection confidence (below this is discarded) |

---

## Why expand_bboxes

Face bounding boxes are small compared to full-body detections. When a face moves between frames, the IoU between its bounding boxes in consecutive frames drops quickly. This causes the tracker to lose the identity and assign a new ID.

`expand_bboxes` scales bounding boxes from their center, adding surrounding context (head, neck, shoulders). Larger boxes overlap more between frames, so the tracker maintains IDs more reliably.

```python
from uniface.tracking import expand_bboxes

# scale=1.0: no expansion (original face bbox)
# scale=1.5: 50% larger (includes some head/neck context)
# scale=2.0: double size (includes upper body)
expanded = expand_bboxes(dets, scale=1.5, image_shape=(height, width))
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `detections` | `np.ndarray` | Shape (N, 5) with `[x1, y1, x2, y2, score]` |
| `scale` | `float` | Expansion factor. 1.0 = no change, 1.5 = 50% larger |
| `image_shape` | `tuple[int, int] \| None` | `(height, width)` for clamping. None to skip clamping |

**Returns:** Array of shape (N, 5) with expanded bboxes and original scores.

---

## Input / Output Format

**Input to `tracker.update()`:**

```python
# (N, 5) array: [x1, y1, x2, y2, confidence]
detections = np.array([
    [100, 50, 200, 160, 0.95],
    [300, 80, 380, 200, 0.87],
])
```

**Output from `tracker.update()`:**

```python
# (M, 5) array: [x1, y1, x2, y2, track_id]
tracks = tracker.update(detections)
# array([[101.2, 51.3, 199.8, 159.8, 1.],
#        [300.5, 80.2, 379.7, 200, 2.]])
```

Track IDs are integers that persist across frames for the same object. The bounding box coordinates come from the Kalman filter prediction, so they may differ slightly from the input detection.

---

## Resetting the Tracker

Call `reset()` when switching to a new video or scene:

```python
tracker.reset()  # Clears all tracks, resets frame counter and ID counter
```

---

## Visualization

`draw_tracks` draws bounding boxes color-coded by track ID, with an ID label above each face:

```python
from uniface.draw import draw_tracks

draw_tracks(
    image=frame,
    faces=faces,
    draw_landmarks=True,     # Draw 5-point landmarks
    draw_id=True,            # Draw "ID:N" label
    corner_bbox=True,        # Corner-style bounding boxes
)
```

---

## CLI Tool

```bash
# Basic tracking
python tools/track.py --source video.mp4

# With bbox expansion
python tools/track.py --source video.mp4 --bbox-scale 1.5

# Webcam
python tools/track.py --source 0 --bbox-scale 1.5

# Save output
python tools/track.py --source video.mp4 --output tracked.mp4 --bbox-scale 1.5

# Use SCRFD detector (default) or RetinaFace
python tools/track.py --source video.mp4 --detector retinaface
```

---

## Next Steps

- [Detection](detection.md) - Face detection models
- [Video & Webcam](../recipes/video-webcam.md) - Video processing patterns
- [Inputs & Outputs](../concepts/inputs-outputs.md) - Data types reference
