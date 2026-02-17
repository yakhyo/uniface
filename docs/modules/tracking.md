# Tracking

Multi-object tracking using [BYTETracker](https://github.com/yakhyo/bytetrack-tracker) with Kalman filtering and IoU-based association. The tracker assigns persistent IDs to detected objects across video frames using a two-stage association strategy — first matching high-confidence detections, then low-confidence ones.

---

## How It Works

BYTETracker takes detection bounding boxes as input and returns tracked bounding boxes with persistent IDs. It does not depend on any specific detector — any source of `[x1, y1, x2, y2, score]` arrays will work.

Each frame, the tracker:

1. Splits detections into high-confidence and low-confidence groups
2. Matches high-confidence detections to existing tracks using IoU
3. Matches remaining tracks to low-confidence detections (second chance)
4. Starts new tracks for unmatched high-confidence detections
5. Removes tracks that have been lost for too long

The Kalman filter predicts where each track will be in the next frame, which helps maintain associations even when detections are noisy.

---

## Basic Usage

```python
import cv2
import numpy as np
from uniface.common import xyxy_to_cxcywh
from uniface.detection import SCRFD
from uniface.tracking import BYTETracker
from uniface.draw import draw_tracks

detector = SCRFD()
tracker = BYTETracker(track_thresh=0.5, track_buffer=30)

cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Detect faces
    faces = detector.detect(frame)
    
    # 2. Track the faces
    tracked_faces = tracker.update_with_faces(faces)   

    # 3. Draw
    tracked_faces = [f for f in tracked_faces if f.track_id is not None]
    draw_tracks(image=frame, faces=tracked_faces)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

Each track ID gets a deterministic color via golden-ratio hue stepping, so the same person keeps the same color across the entire video.

---

## Webcam Tracking

```python
import cv2
import numpy as np
from uniface.common import xyxy_to_cxcywh
from uniface.detection import SCRFD
from uniface.tracking import BYTETracker
from uniface.draw import draw_tracks

detector = SCRFD()
tracker = BYTETracker(track_thresh=0.5, track_buffer=30)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector.detect(frame)
    tracked_faces = tracker.update_with_faces(faces)   

    tracked_faces = [f for f in tracked_faces if f.track_id is not None]
    cv2.imshow("Face Tracking - Press 'q' to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Parameters

```python
from uniface.tracking import BYTETracker

tracker = BYTETracker(
    track_thresh=0.5,
    track_buffer=30,
    match_thresh=0.8,
    low_thresh=0.1,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `track_thresh` | 0.5 | Detections above this score go through first-pass association |
| `track_buffer` | 30 | How many frames to keep a lost track before removing it |
| `match_thresh` | 0.8 | IoU threshold for matching tracks to detections |
| `low_thresh` | 0.1 | Detections below this score are discarded entirely |

---

## Input / Output

**Input** — list of detected faces.

**Output** — list of tracked faces.

The output bounding box for each face come from the Kalman filter prediction, so they may differ slightly from the input. Track IDs are integers that persist across frames for the same object.

---

## Resetting the Tracker

When switching to a different video or scene, reset the tracker to clear all internal state:

```python
tracker.reset()
```

This clears all active, lost, and removed tracks, resets the frame counter, and resets the ID counter back to zero.

---

## Visualization

`draw_tracks` draws bounding boxes color-coded by track ID:

```python
from uniface.draw import draw_tracks

draw_tracks(
    image=frame,
    faces=tracked_faces,
    draw_landmarks=True,
    draw_id=True,
    corner_bbox=True,
)
```

---

## Small Face Performance

!!! warning "Tracking performance with small faces"
    The tracker relies on IoU (Intersection over Union) to match detections across
    frames. When faces occupy a small portion of the image — for example in
    surveillance footage or wide-angle cameras — even slight movement between frames
    can cause a large drop in IoU. This makes it harder for the tracker to maintain
    consistent IDs, and you may see IDs switching or resetting more often than expected.

    This is not specific to BYTETracker; it applies to any IoU-based tracker. A few
    things that can help:

    - **Lower `match_thresh`** (e.g. `0.5` or `0.6`) so the tracker accepts lower
      overlap as a valid match.
    - **Increase `track_buffer`** (e.g. `60` or higher) to hold onto lost tracks
      longer before discarding them.
    - **Use a higher-resolution input** if possible, so face bounding boxes are
      larger in pixel terms.

    ```python
    tracker = BYTETracker(
        track_thresh=0.4,
        track_buffer=60,
        match_thresh=0.6,
    )
    ```

---

## CLI Tool

```bash
# Track faces in a video
python tools/track.py --source video.mp4

# Webcam
python tools/track.py --source 0

# Save output
python tools/track.py --source video.mp4 --output tracked.mp4

# Use RetinaFace instead of SCRFD
python tools/track.py --source video.mp4 --detector retinaface

# Keep lost tracks longer
python tools/track.py --source video.mp4 --track-buffer 60
```

---

## References

- [yakhyo/bytetrack-tracker](https://github.com/yakhyo/bytetrack-tracker) — standalone BYTETracker implementation used in UniFace
- [ByteTrack paper](https://arxiv.org/abs/2110.06864) — Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"

---

## See Also

- [Detection](detection.md) — face detection models
- [Video & Webcam](../recipes/video-webcam.md) — video processing patterns
- [Inputs & Outputs](../concepts/inputs-outputs.md) — data types and formats
