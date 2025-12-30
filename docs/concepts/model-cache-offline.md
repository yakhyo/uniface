# Model Cache & Offline Use

UniFace automatically downloads and caches models. This page explains how model management works.

---

## Automatic Download

Models are downloaded on first use:

```python
from uniface import RetinaFace

# First run: downloads model to cache
detector = RetinaFace()  # ~3.5 MB download

# Subsequent runs: loads from cache
detector = RetinaFace()  # Instant
```

---

## Cache Location

Default cache directory:

```
~/.uniface/models/
```

**Example structure:**

```
~/.uniface/models/
├── retinaface_mv2.onnx
├── w600k_mbf.onnx
├── 2d106det.onnx
├── gaze_resnet34.onnx
├── parsing_resnet18.onnx
└── ...
```

---

## Custom Cache Directory

Specify a custom cache location:

```python
from uniface.model_store import verify_model_weights
from uniface.constants import RetinaFaceWeights

# Download to custom directory
model_path = verify_model_weights(
    RetinaFaceWeights.MNET_V2,
    root='./my_models'
)
print(f"Model at: {model_path}")
```

---

## Pre-Download Models

Download models before deployment:

```python
from uniface.model_store import verify_model_weights
from uniface.constants import (
    RetinaFaceWeights,
    ArcFaceWeights,
    AgeGenderWeights,
)

# Download all needed models
models = [
    RetinaFaceWeights.MNET_V2,
    ArcFaceWeights.MNET,
    AgeGenderWeights.DEFAULT,
]

for model in models:
    path = verify_model_weights(model)
    print(f"Downloaded: {path}")
```

Or use the CLI tool:

```bash
python tools/download_model.py
```

---

## Offline Use

For air-gapped or offline environments:

### 1. Pre-download models

On a connected machine:

```python
from uniface.model_store import verify_model_weights
from uniface.constants import RetinaFaceWeights

path = verify_model_weights(RetinaFaceWeights.MNET_V2)
print(f"Copy from: {path}")
```

### 2. Copy to target machine

```bash
# Copy the entire cache directory
scp -r ~/.uniface/models/ user@offline-machine:~/.uniface/models/
```

### 3. Use normally

```python
# Models load from local cache
from uniface import RetinaFace
detector = RetinaFace()  # No network required
```

---

## Model Verification

Models are verified with SHA-256 checksums:

```python
from uniface.constants import MODEL_SHA256, RetinaFaceWeights

# Check expected checksum
expected = MODEL_SHA256[RetinaFaceWeights.MNET_V2]
print(f"Expected SHA256: {expected}")
```

If a model fails verification, it's re-downloaded automatically.

---

## Available Models

### Detection Models

| Model | Size | Download |
|-------|------|----------|
| RetinaFace MNET_025 | 1.7 MB | ✅ |
| RetinaFace MNET_V2 | 3.5 MB | ✅ |
| RetinaFace RESNET34 | 56 MB | ✅ |
| SCRFD 500M | 2.5 MB | ✅ |
| SCRFD 10G | 17 MB | ✅ |
| YOLOv5n-Face | 11 MB | ✅ |
| YOLOv5s-Face | 28 MB | ✅ |
| YOLOv5m-Face | 82 MB | ✅ |

### Recognition Models

| Model | Size | Download |
|-------|------|----------|
| ArcFace MNET | 8 MB | ✅ |
| ArcFace RESNET | 166 MB | ✅ |
| MobileFace MNET_V2 | 4 MB | ✅ |
| SphereFace SPHERE20 | 50 MB | ✅ |

### Other Models

| Model | Size | Download |
|-------|------|----------|
| Landmark106 | 14 MB | ✅ |
| AgeGender | 8 MB | ✅ |
| FairFace | 44 MB | ✅ |
| Gaze ResNet34 | 82 MB | ✅ |
| BiSeNet ResNet18 | 51 MB | ✅ |
| MiniFASNet V2 | 1.2 MB | ✅ |

---

## Clear Cache

Remove cached models:

```bash
# Remove all cached models
rm -rf ~/.uniface/models/

# Remove specific model
rm ~/.uniface/models/retinaface_mv2.onnx
```

Models will be re-downloaded on next use.

---

## Environment Variables

Set custom cache location via environment variable:

```bash
export UNIFACE_CACHE_DIR=/path/to/custom/cache
```

```python
import os
os.environ['UNIFACE_CACHE_DIR'] = '/path/to/custom/cache'

from uniface import RetinaFace
detector = RetinaFace()  # Uses custom cache
```

---

## Next Steps

- [Thresholds & Calibration](thresholds-calibration.md) - Tune model parameters
- [Detection Module](../modules/detection.md) - Detection model details
