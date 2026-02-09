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
├── retinaface_mnet_v2.onnx
├── arcface_mnet.onnx
├── 2d_106.onnx
├── gaze_resnet34.onnx
├── parsing_resnet18.onnx
└── ...
```

---

## Custom Cache Directory

Use the programmatic API to change the cache location at runtime:

```python
from uniface import set_cache_dir, get_cache_dir

# Set a custom cache directory
set_cache_dir('/data/models')

# Verify the current path
print(get_cache_dir())  # /data/models

# All subsequent model loads use the new directory
from uniface import RetinaFace
detector = RetinaFace()  # Downloads to /data/models/
```

Or set the `UNIFACE_CACHE_DIR` environment variable (see [Environment Variables](#environment-variables) below).

---

## Pre-Download Models

Download models before deployment using the concurrent downloader:

```python
from uniface import download_models
from uniface.constants import (
    RetinaFaceWeights,
    ArcFaceWeights,
    AgeGenderWeights,
)

# Download multiple models concurrently (up to 4 threads by default)
paths = download_models([
    RetinaFaceWeights.MNET_V2,
    ArcFaceWeights.MNET,
    AgeGenderWeights.DEFAULT,
])

for model, path in paths.items():
    print(f"{model.value} -> {path}")
```

Or download one at a time:

```python
from uniface.model_store import verify_model_weights
from uniface.constants import RetinaFaceWeights

path = verify_model_weights(RetinaFaceWeights.MNET_V2)
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

### 3. Point to the cache (if non-default location)

```python
from uniface import set_cache_dir

# Only needed if the models are not at ~/.uniface/models/
set_cache_dir('/path/to/copied/models')
```

### 4. Use normally

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
| YOLOv8-Lite-S | 7.4 MB | ✅ |
| YOLOv8n-Face | 12 MB | ✅ |

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

Find and remove cached models:

```python
from uniface import get_cache_dir
print(get_cache_dir())  # shows the active cache path
```

```bash
# Remove all cached models
rm -rf ~/.uniface/models/

# Remove specific model
rm ~/.uniface/models/retinaface_mv2.onnx
```

Models will be re-downloaded on next use.

---

## Environment Variables

There are three equivalent ways to configure the cache directory:

**1. Programmatic API (recommended)**

```python
from uniface import set_cache_dir, get_cache_dir

set_cache_dir('/path/to/custom/cache')
print(get_cache_dir())  # /path/to/custom/cache
```

**2. Direct environment variable (Python)**

```python
import os
os.environ['UNIFACE_CACHE_DIR'] = '/path/to/custom/cache'

from uniface import RetinaFace
detector = RetinaFace()  # Uses custom cache
```

**3. Shell environment variable**

```bash
export UNIFACE_CACHE_DIR=/path/to/custom/cache
```

All three methods set the same `UNIFACE_CACHE_DIR` environment variable under the hood. `get_cache_dir()` always returns the resolved path.

---

## Next Steps

- [Thresholds & Calibration](thresholds-calibration.md) - Tune model parameters
- [Detection Module](../modules/detection.md) - Detection model details
