# Video & Webcam

Real-time face analysis for video streams.

!!! note "Work in Progress"
    This page contains example code patterns. Test thoroughly before using in production.

---

## Webcam Detection

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

    draw_detections(
        image=frame,
        bboxes=[f.bbox for f in faces],
        scores=[f.confidence for f in faces],
        landmarks=[f.landmarks for f in faces]
    )

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Video File Processing

```python
import cv2
from uniface import RetinaFace

def process_video(input_path, output_path):
    """Process a video file."""
    detector = RetinaFace()
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.read()[0]:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect(frame)
        # ... process and draw ...

        out.write(frame)

    cap.release()
    out.release()

# Usage
process_video("input.mp4", "output.mp4")
```

---

## Performance Tips

### Skip Frames

```python
PROCESS_EVERY_N = 3  # Process every 3rd frame
frame_count = 0
last_faces = []

while True:
    ret, frame = cap.read()
    if frame_count % PROCESS_EVERY_N == 0:
        last_faces = detector.detect(frame)
    frame_count += 1
    # Draw last_faces...
```

### FPS Counter

```python
import time

prev_time = time.time()
while True:
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
```

---

## See Also

- [Anonymize Stream](anonymize-stream.md) - Privacy protection in video
- [Batch Processing](batch-processing.md) - Process multiple files
- [Detection Module](../modules/detection.md) - Detection options
- [Gaze Module](../modules/gaze.md) - Gaze tracking
