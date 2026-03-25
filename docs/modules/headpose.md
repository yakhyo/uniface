# Head Pose Estimation

Head pose estimation predicts the 3D orientation of a person's head (pitch, yaw, and roll angles).

---

## Available Models

| Model | Backbone | Size | MAE* |
|-------|----------|------|------|
| **ResNet18** :material-check-circle: | ResNet18 | 43 MB | 5.22° |
| ResNet34 | ResNet34 | 82 MB | 5.07° |
| ResNet50 | ResNet50 | 91 MB | 4.83° |
| MobileNetV2 | MobileNetV2 | 9.6 MB | 5.72° |
| MobileNetV3-Small | MobileNetV3 | 4.8 MB | 6.31° |
| MobileNetV3-Large | MobileNetV3 | 16 MB | 5.58° |

*MAE = Mean Absolute Error on AFLW2000 test set (lower is better)

---

## Basic Usage

```python
import cv2
from uniface.detection import RetinaFace
from uniface.headpose import HeadPose

detector = RetinaFace()
head_pose = HeadPose()

image = cv2.imread("photo.jpg")
faces = detector.detect(image)

for face in faces:
    # Crop face
    x1, y1, x2, y2 = map(int, face.bbox)
    face_crop = image[y1:y2, x1:x2]

    if face_crop.size > 0:
        # Estimate head pose
        result = head_pose.estimate(face_crop)
        print(f"Pitch: {result.pitch:.1f}°, Yaw: {result.yaw:.1f}°, Roll: {result.roll:.1f}°")
```

---

## Model Variants

```python
from uniface.headpose import HeadPose
from uniface.constants import HeadPoseWeights

# Default (ResNet18, recommended balance of speed and accuracy)
hp = HeadPose()

# Lightweight for mobile/edge
hp = HeadPose(model_name=HeadPoseWeights.MOBILENET_V3_SMALL)

# Higher accuracy
hp = HeadPose(model_name=HeadPoseWeights.RESNET50)
```

---

## Output Format

```python
result = head_pose.estimate(face_crop)

# HeadPoseResult dataclass
result.pitch  # Rotation around X-axis in degrees
result.yaw    # Rotation around Y-axis in degrees
result.roll   # Rotation around Z-axis in degrees
```

### Angle Convention

```
          pitch > 0 (looking down)
               │
               │
yaw < 0  ─────┼───── yaw > 0
(looking left) │     (looking right)
               │
          pitch < 0 (looking up)

roll > 0 = clockwise tilt
roll < 0 = counter-clockwise tilt
```

- **Pitch**: Rotation around X-axis (positive = looking down)
- **Yaw**: Rotation around Y-axis (positive = looking right)
- **Roll**: Rotation around Z-axis (positive = tilting clockwise)

---

## Visualization

### 3D Cube (default)

The default visualization draws a wireframe cube oriented to match the head pose.

```python
from uniface.draw import draw_head_pose

faces = detector.detect(image)

for face in faces:
    x1, y1, x2, y2 = map(int, face.bbox)
    face_crop = image[y1:y2, x1:x2]

    if face_crop.size > 0:
        result = head_pose.estimate(face_crop)

        # Draw cube on image (default)
        draw_head_pose(image, face.bbox, result.pitch, result.yaw, result.roll)

cv2.imwrite("headpose_output.jpg", image)
```

### Axis Visualization

```python
from uniface.draw import draw_head_pose

# X/Y/Z coordinate axes
draw_head_pose(image, face.bbox, result.pitch, result.yaw, result.roll, draw_type='axis')
```

### Low-Level Drawing Functions

```python
from uniface.draw import draw_head_pose_cube, draw_head_pose_axis

# Draw cube directly
draw_head_pose_cube(image, yaw=10.0, pitch=-5.0, roll=2.0, bbox=[100, 100, 250, 280])

# Draw axes directly
draw_head_pose_axis(image, yaw=10.0, pitch=-5.0, roll=2.0, bbox=[100, 100, 250, 280])
```

---

## Real-Time Head Pose Tracking

```python
import cv2
from uniface.detection import RetinaFace
from uniface.headpose import HeadPose
from uniface.draw import draw_head_pose

detector = RetinaFace()
head_pose = HeadPose()

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
            result = head_pose.estimate(face_crop)
            draw_head_pose(frame, face.bbox, result.pitch, result.yaw, result.roll)

    cv2.imshow("Head Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Use Cases

### Driver Drowsiness Detection

```python
def is_head_drooping(result, pitch_threshold=-15):
    """Check if the head is drooping (looking down significantly)."""
    return result.pitch < pitch_threshold

result = head_pose.estimate(face_crop)
if is_head_drooping(result):
    print("Warning: Head drooping detected")
```

### Attention Monitoring

```python
def is_facing_forward(result, threshold=20):
    """Check if the person is facing roughly forward."""
    return (
        abs(result.pitch) < threshold
        and abs(result.yaw) < threshold
        and abs(result.roll) < threshold
    )

result = head_pose.estimate(face_crop)
if is_facing_forward(result):
    print("Facing forward")
else:
    print("Looking away")
```

---

## Factory Function

```python
from uniface.headpose import create_head_pose_estimator

hp = create_head_pose_estimator()  # Returns HeadPose
```

---

## Next Steps

- [Gaze Estimation](gaze.md) - Eye gaze direction
- [Anti-Spoofing](spoofing.md) - Face liveness detection
- [Video Recipe](../recipes/video-webcam.md) - Real-time processing
