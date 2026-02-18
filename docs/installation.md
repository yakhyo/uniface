# Installation

This guide covers all installation options for UniFace.

---

## Requirements

- **Python**: 3.10 or higher
- **Operating Systems**: macOS, Linux, Windows

---

## Quick Install

The simplest way to install UniFace:

```bash
pip install uniface
```

This installs the CPU version with all core dependencies.

---

## Platform-Specific Installation

### macOS (Apple Silicon - M1/M2/M3/M4)

For Apple Silicon Macs, the standard installation automatically includes ARM64 optimizations:

```bash
pip install uniface
```

!!! tip "Native Performance"
    The base `onnxruntime` package has native Apple Silicon support with ARM64 optimizations built-in since version 1.13+. No additional configuration needed.

Verify ARM64 installation:

```bash
python -c "import platform; print(platform.machine())"
# Should show: arm64
```

---

### Linux/Windows with NVIDIA GPU

For CUDA acceleration on NVIDIA GPUs:

```bash
pip install uniface[gpu]
```

**Requirements:**

- CUDA 11.x or 12.x
- cuDNN 8.x

!!! info "CUDA Compatibility"
    See [ONNX Runtime GPU requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) for detailed compatibility matrix.

Verify GPU installation:

```python
import onnxruntime as ort
print("Available providers:", ort.get_available_providers())
# Should include: 'CUDAExecutionProvider'
```

---

### FAISS Vector Indexing

For fast multi-identity face search using a FAISS index:

```bash
pip install faiss-cpu   # CPU
pip install faiss-gpu   # NVIDIA GPU (CUDA)
```

See the [Indexing module](modules/indexing.md) for usage.

---

### CPU-Only (All Platforms)

```bash
pip install uniface
```

Works on all platforms with automatic CPU fallback.

---

## Install from Source

For development or the latest features:

```bash
git clone https://github.com/yakhyo/uniface.git
cd uniface
pip install -e .
```

With development dependencies:

```bash
pip install -e ".[dev]"
```

---

## Dependencies

UniFace has minimal dependencies:

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations |
| `opencv-python` | Image processing |
| `onnxruntime` | Model inference |
| `scikit-image` | Geometric transforms |
| `requests` | Model download |
| `tqdm` | Progress bars |

**Optional:**

| Package | Install extra | Purpose |
|---------|---------------|---------|
| `faiss-cpu` / `faiss-gpu` | `pip install faiss-cpu` | FAISS vector indexing |
| `onnxruntime-gpu` | `uniface[gpu]` | CUDA acceleration |

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

## Troubleshooting

### Import Errors

If you encounter import errors, ensure you're using Python 3.10+:

```bash
python --version
# Should show: Python 3.10.x or higher
```

### Model Download Issues

Models are automatically downloaded on first use. If downloads fail:

```python
from uniface.model_store import verify_model_weights
from uniface.constants import RetinaFaceWeights

# Manually download a model
model_path = verify_model_weights(RetinaFaceWeights.MNET_V2)
print(f"Model downloaded to: {model_path}")
```

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
