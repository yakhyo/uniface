# Video & Webcam

Real-time face analysis for video streams.

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

    # Detect faces
    faces = detector.detect(frame)

    # Draw results
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
from uniface.visualization import draw_detections

def process_video(input_path, output_path):
    """Process a video file."""
    detector = RetinaFace()

    cap = cv2.VideoCapture(input_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and draw
        faces = detector.detect(frame)
        draw_detections(
            image=frame,
            bboxes=[f.bbox for f in faces],
            scores=[f.confidence for f in faces],
            landmarks=[f.landmarks for f in faces]
        )

        out.write(frame)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    cap.release()
    out.release()
    print(f"Saved to {output_path}")

# Usage
process_video("input.mp4", "output.mp4")
```

---

## FPS Counter

Add frame rate display:

```python
import cv2
import time
from uniface import RetinaFace

detector = RetinaFace()
cap = cv2.VideoCapture(0)

prev_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Detect faces
    faces = detector.detect(frame)

    # Draw FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw detections
    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Skip Frames for Performance

Process every N frames for better performance:

```python
import cv2
from uniface import RetinaFace

detector = RetinaFace()
cap = cv2.VideoCapture(0)

PROCESS_EVERY_N = 3  # Process every 3rd frame
frame_count = 0
last_faces = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Only detect every N frames
    if frame_count % PROCESS_EVERY_N == 0:
        last_faces = detector.detect(frame)

    # Draw last detection results
    for face in last_faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Full Analysis Pipeline

Real-time detection with age/gender:

```python
import cv2
from uniface import RetinaFace, AgeGender

detector = RetinaFace()
age_gender = AgeGender()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector.detect(frame)

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Predict age/gender
        result = age_gender.predict(frame, face.bbox)
        label = f"{result.sex}, {result.age}y"

        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Age/Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Gaze Tracking

Real-time gaze estimation:

```python
import cv2
import numpy as np
from uniface import RetinaFace, MobileGaze
from uniface.visualization import draw_gaze

detector = RetinaFace()
gaze = MobileGaze()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector.detect(frame)

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size > 0:
            result = gaze.estimate(face_crop)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw gaze arrow
            draw_gaze(frame, face.bbox, result.pitch, result.yaw)

            # Display angles
            label = f"P:{np.degrees(result.pitch):.0f} Y:{np.degrees(result.yaw):.0f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Gaze Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Recording Output

Record processed video:

```python
import cv2
from uniface import RetinaFace

detector = RetinaFace()
cap = cv2.VideoCapture(0)

# Get camera properties
fps = 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Setup recording
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('recording.mp4', fourcc, fps, (width, height))

is_recording = False

print("Press 'r' to start/stop recording, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector.detect(frame)

    # Draw detections
    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Recording indicator
    if is_recording:
        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
        out.write(frame)

    cv2.imshow("Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        is_recording = not is_recording
        print(f"Recording: {is_recording}")
    elif key == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

---

## Multi-Camera

Process multiple cameras:

```python
import cv2
from uniface import RetinaFace

detector = RetinaFace()

# Open multiple cameras
caps = [
    cv2.VideoCapture(0),
    cv2.VideoCapture(1)  # Second camera
]

while True:
    frames = []

    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if ret:
            faces = detector.detect(frame)

            for face in faces:
                x1, y1, x2, y2 = map(int, face.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            frames.append(frame)

    # Display side by side
    if len(frames) == 2:
        combined = cv2.hconcat(frames)
        cv2.imshow("Multi-Camera", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
```

---

## Next Steps

- [Anonymize Stream](anonymize-stream.md) - Privacy in video
- [Face Search](face-search.md) - Identity search
- [Image Pipeline](image-pipeline.md) - Full analysis
