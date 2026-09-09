# Portrait Matting

Portrait matting produces a soft alpha matte separating the foreground (person) from the background — no trimap needed.

<figure markdown="span">
  ![Portrait Matting](https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/matting.jpg){ width="100%" }
  <figcaption>MODNet: Input → Matte → Green Screen</figcaption>
</figure>

---

## Available Models

| Model | Variant | Size | Use Case |
|-------|---------|------|----------|
| **MODNet Photographic** :material-check-circle: | PHOTOGRAPHIC | 25 MB | High-quality portrait photos |
| MODNet Webcam | WEBCAM | 25 MB | Real-time webcam feeds |
| **RobustVideoMatting** :material-check-circle: | MOBILENETV3 | 15 MB | Temporal-stable matting for video |
| RobustVideoMatting | RESNET50 | 107 MB | Video matting with a higher-capacity backbone |

---

## Basic Usage

```python
import cv2
from uniface.matting import MODNet

matting = MODNet()

image = cv2.imread("photo.jpg")
matte = matting.predict(image)

print(f"Matte shape: {matte.shape}")   # (H, W)
print(f"Matte dtype: {matte.dtype}")   # float32
print(f"Matte range: [{matte.min():.2f}, {matte.max():.2f}]")  # [0, 1]
```

---

## Model Variants

```python
from uniface.matting import MODNet
from uniface.constants import MODNetWeights

# Photographic (default) — best for photos
matting = MODNet()

# Webcam — optimized for real-time
matting = MODNet(model_name=MODNetWeights.WEBCAM)

# Custom input size
matting = MODNet(input_size=256)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `PHOTOGRAPHIC` | Model variant to load |
| `input_size` | `512` | Target shorter-side size for preprocessing |
| `providers` | `None` | ONNX Runtime execution providers |

---

## Applications

### Transparent Background (RGBA)

```python
import cv2
import numpy as np

matting = MODNet()
image = cv2.imread("photo.jpg")
matte = matting.predict(image)

rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
rgba[:, :, 3] = (matte * 255).astype(np.uint8)
cv2.imwrite("transparent.png", rgba)
```

### Green Screen

```python
import numpy as np

matte_3ch = matte[:, :, np.newaxis]
bg = np.full_like(image, (0, 177, 64), dtype=np.uint8)
green = (image * matte_3ch + bg * (1 - matte_3ch)).astype(np.uint8)
cv2.imwrite("green_screen.jpg", green)
```

<figure markdown="span">
  ![Matting on flyaway hair](https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/matting_alt.jpg){ width="100%" }
  <figcaption>A plain, low-contrast background: individual flyaway strands survive into the composite</figcaption>
</figure>

MODNet is trimap-free, so a background that competes with the subject in sharpness and contrast
is where the alpha edge softens. Compositing onto a blurred or plain background hides most of it;
if you need a clean cut, shoot against a plain wall.

---

## Video Matting with RobustVideoMatting

For **video**, per-frame matting flickers on hair and other high-frequency edges. RobustVideoMatting
(RVM) is a *recurrent* network: it carries temporal memory across frames, so the alpha stays stable
from frame to frame.

```python
import cv2
from uniface.matting import RobustVideoMatting

matting = RobustVideoMatting()          # MobileNetV3 (fast). RESNET50 is the higher-capacity variant.

cap = cv2.VideoCapture("video.mp4")
matting.reset()                          # Start a new sequence (also call after scene cuts)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    matte = matting.predict_frame(frame)  # (H, W) float32 in [0, 1], reusing temporal memory
    # ... composite `frame` and `matte` as shown above ...
```

For a list of frames at once (e.g. a preloaded clip), `predict_frames` returns the stacked mattes:

```python
mattes = matting.predict_frames(frames)  # (T, H, W) float32
```

### API notes

- `predict(image)` processes a single image **without** temporal memory — use it for unrelated stills.
  It never carries state between calls.
- `predict_frame(frame)` / `predict_frames(frames)` reuse memory across frames. Memory is kept
  between calls for chunked processing, and is reset automatically if the frame size changes.
  Call `reset()` before starting an unrelated video.
- `downsample_ratio` trades speed against detail (default `None` = auto, largest side mapped to
  ~512 px). Lower it for lower resolutions, raise it when the full body is in shot.
- The RVM model weights are [GPL-3.0 licensed](../license-attribution.md); check before shipping
  commercially.

```python
from uniface.constants import RobustVideoMattingWeights
from uniface.matting import RobustVideoMatting

matting = RobustVideoMatting(model_name=RobustVideoMattingWeights.RESNET50,
                             downsample_ratio=0.25)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `MOBILENETV3` | Model variant to load |
| `downsample_ratio` | `None` | Internal working resolution fraction; `None` = auto |
| `providers` | `None` | ONNX Runtime execution providers |

### Custom Background

```python
import cv2
import numpy as np

background = cv2.imread("beach.jpg")
background = cv2.resize(background, (image.shape[1], image.shape[0]))

matte_3ch = matte[:, :, np.newaxis]
result = (image * matte_3ch + background * (1 - matte_3ch)).astype(np.uint8)
cv2.imwrite("custom_bg.jpg", result)
```

### Webcam Matting

```python
import cv2
import numpy as np
from uniface.matting import MODNet
from uniface.constants import MODNetWeights

matting = MODNet(model_name=MODNetWeights.WEBCAM)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    matte = matting.predict(frame)
    matte_3ch = matte[:, :, np.newaxis]
    bg = np.full_like(frame, (0, 177, 64), dtype=np.uint8)
    result = (frame * matte_3ch + bg * (1 - matte_3ch)).astype(np.uint8)

    cv2.imshow("Matting", np.hstack([frame, result]))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Available Matting Models

```python
from uniface.constants import MODNetWeights
from uniface.matting import MODNet

# Default (Photographic)
matting = MODNet()

# Webcam variant
matting = MODNet(model_name=MODNetWeights.WEBCAM)
```

---

## Next Steps

- [Parsing](parsing.md) - Face semantic segmentation
- [Privacy](privacy.md) - Face anonymization
- [Detection](detection.md) - Face detection
- [CLI Tools](https://github.com/yakhyo/uniface/blob/main/tools/README.md) - Command-line scripts for all UniFace modules
