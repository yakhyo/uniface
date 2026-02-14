# Anonymize Stream

Blur faces in real-time video streams for privacy protection.

!!! note "Work in Progress"
    This page contains example code patterns. Test thoroughly before using in production.

---

## Webcam Anonymization

```python
import cv2
from uniface.detection import RetinaFace
from uniface.privacy import BlurFace

detector = RetinaFace()
blurrer = BlurFace(method='pixelate')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector.detect(frame)
    frame = blurrer.anonymize(frame, faces, inplace=True)

    cv2.imshow('Anonymized', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Video File Anonymization

```python
import cv2
from uniface.detection import RetinaFace
from uniface.privacy import BlurFace

detector = RetinaFace()
blurrer = BlurFace(method='gaussian')

cap = cv2.VideoCapture("input.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(3)), int(cap.get(4))

out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

while cap.read()[0]:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector.detect(frame)
    blurrer.anonymize(frame, faces, inplace=True)
    out.write(frame)

cap.release()
out.release()
```

---

## Single Image

```python
import cv2
from uniface.detection import RetinaFace
from uniface.privacy import BlurFace

detector = RetinaFace()
blurrer = BlurFace(method='pixelate')

image = cv2.imread("photo.jpg")
faces = detector.detect(image)
result = blurrer.anonymize(image, faces)
cv2.imwrite("anonymized.jpg", result)
```

---

## Available Blur Methods

| Method | Usage |
|--------|-------|
| Pixelate | `BlurFace(method='pixelate', pixel_blocks=15)` |
| Gaussian | `BlurFace(method='gaussian', blur_strength=3.0)` |
| Blackout | `BlurFace(method='blackout', color=(0,0,0))` |
| Elliptical | `BlurFace(method='elliptical', margin=20)` |
| Median | `BlurFace(method='median', blur_strength=3.0)` |

---

## See Also

- [Privacy Module](../modules/privacy.md) - Privacy protection details
- [Video & Webcam](video-webcam.md) - Real-time processing
- [Detection Module](../modules/detection.md) - Face detection
