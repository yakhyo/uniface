# Installation

This guide covers all installation options for UniFace.

---

## Requirements

- **Python**: 3.10 or higher
- **Operating Systems**: macOS, Linux, Windows

---

## Why Two Extras?

`onnxruntime` (CPU) and `onnxruntime-gpu` (CUDA) both own the same Python namespace.
Installing both at the same time causes file conflicts and silent provider mismatches.
UniFace exposes them as separate, mutually exclusive extras so you install exactly one.

---

## Quick Install

=== "CPU / Apple Silicon"

    ```bash
    pip install uniface[cpu]
    ```

=== "NVIDIA GPU (CUDA)"

    ```bash
    pip install uniface[gpu]
    ```

---

## Platform-Specific Installation

### macOS (Apple Silicon - M1/M2/M3/M4)

The `[cpu]` extra pulls in the standard `onnxruntime` package, which has native ARM64 support
built in since version 1.13. No additional setup is needed for CoreML acceleration.

```bash
pip install uniface[cpu]
```

!!! tip "Native Performance"
    `onnxruntime` 1.13+ includes ARM64 optimizations out of the box.
    UniFace automatically detects and enables `CoreMLExecutionProvider` on Apple Silicon.

Verify ARM64 installation:

```bash
python -c "import platform; print(platform.machine())"
# Should show: arm64
```

---

### Linux/Windows with NVIDIA GPU

```bash
pip install uniface[gpu]
```

This installs `onnxruntime-gpu`, which includes both `CUDAExecutionProvider` and
`CPUExecutionProvider` — no separate CPU package is needed.

**Requirements:**

- NVIDIA driver compatible with your CUDA version
- CUDA 11.x or 12.x toolkit
- cuDNN 8.x

!!! info "CUDA Compatibility"
    See the [ONNX Runtime GPU compatibility matrix](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
    for matching CUDA and cuDNN versions.

Verify GPU installation:

```python
import onnxruntime as ort
print("Available providers:", ort.get_available_providers())
# Should include: 'CUDAExecutionProvider'
```

---

### CPU-Only (All Platforms)

```bash
pip install uniface[cpu]
```

Works on all platforms with automatic CPU fallback.

---

## Install from Source

For development or the latest features:

```bash
git clone https://github.com/yakhyo/uniface.git
cd uniface

pip install -e ".[cpu]"   # CPU / Apple Silicon
pip install -e ".[gpu]"   # NVIDIA GPU
```

With development dependencies:

```bash
pip install -e ".[cpu,dev]"
```

---

## FAISS Vector Store

For fast multi-identity face search using a FAISS vector store:

```bash
pip install faiss-cpu   # CPU
pip install faiss-gpu   # NVIDIA GPU (CUDA)
```

See the [Stores module](modules/stores.md) for usage.

---

## Dependencies

UniFace has minimal core dependencies:

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations |
| `opencv-python` | Image processing |
| `scikit-image` | Geometric transforms |
| `scipy` | Signal processing |
| `requests` | Model download |
| `tqdm` | Progress bars |

**Runtime extras (install exactly one):**

| Extra | Package | Use case |
|-------|---------|---------|
| `uniface[cpu]` | `onnxruntime` | CPU inference, Apple Silicon |
| `uniface[gpu]` | `onnxruntime-gpu` | NVIDIA CUDA inference |

**Other optional packages:**

| Package | Install | Purpose |
|---------|---------|---------|
| `faiss-cpu` / `faiss-gpu` | `pip install faiss-cpu` | FAISS vector store |
| `torch` | `pip install torch` | Emotion model (TorchScript) |
| `torchvision` | `pip install torchvision` | Faster NMS for YOLO detectors |

---

## Verify Installation

Test your installation:

```python
import uniface
print(f"UniFace version: {uniface.__version__}")

# Check available ONNX providers
import onnxruntime as ort
print(f"Available providers: {ort.get_available_providers()}")

# Quick test
from uniface.detection import RetinaFace
detector = RetinaFace()
print("Installation successful!")
```

---

## Upgrading

When upgrading UniFace, stay consistent with your runtime extra:

```bash
pip install --upgrade uniface[cpu]   # or uniface[gpu]
```

If you are switching from CPU to GPU (or vice versa):

```bash
pip uninstall onnxruntime onnxruntime-gpu -y
pip install uniface[gpu]   # install the one you want
```

---

## Pre-release Versions

UniFace ships release candidates and betas to PyPI ahead of stable releases (versions like `0.7.0rc1`, `0.7.0b1`, `0.7.0a1`). These let you try upcoming features before they're finalized.

`pip install uniface` always installs the latest **stable** release. To opt in to pre-releases:

```bash
# Latest pre-release (if newer than latest stable)
pip install uniface[cpu] --pre

# A specific pre-release
pip install uniface[cpu]==0.7.0rc1
```

Pre-releases are not recommended for production — APIs may still change before the stable release.

---

## Troubleshooting

### onnxruntime Not Found

If you see:

```
ImportError: onnxruntime is not installed. Install it with one of:
  pip install uniface[cpu]   # CPU / Apple Silicon
  pip install uniface[gpu]   # NVIDIA GPU (CUDA)
```

You installed uniface without an extra. Run the appropriate command above.

---

### Both onnxruntime and onnxruntime-gpu Installed

If you previously ran `pip install uniface[gpu]` on top of a `pip install uniface[cpu]`
(or vice versa), you may have both packages installed simultaneously, which causes conflicts.
Fix it with:

```bash
pip uninstall onnxruntime onnxruntime-gpu -y
pip install uniface[gpu]   # or uniface[cpu]
```

---

### Import Errors

Ensure you're using Python 3.10+:

```bash
python --version
# Should show: Python 3.10.x or higher
```

---

### Model Download Issues

Models are automatically downloaded on first use. If downloads fail:

```python
from uniface.model_store import verify_model_weights
from uniface.constants import RetinaFaceWeights

# Manually download a model
model_path = verify_model_weights(RetinaFaceWeights.MNET_V2)
print(f"Model downloaded to: {model_path}")
```

---

### CUDA Not Detected

1. Verify CUDA installation:
   ```bash
   nvidia-smi
   ```

2. Check CUDA version compatibility with ONNX Runtime.

3. Reinstall the GPU extra cleanly:
   ```bash
   pip uninstall onnxruntime onnxruntime-gpu -y
   pip install uniface[gpu]
   ```

---

### Performance Issues on Mac

Verify you're using the ARM64 build (not x86_64 via Rosetta):

```bash
python -c "import platform; print(platform.machine())"
# Should show: arm64 (not x86_64)
```

---

## Next Steps

- [Quickstart Guide](quickstart.md) - Get started in 5 minutes
- [Execution Providers](concepts/execution-providers.md) - Hardware acceleration setup
