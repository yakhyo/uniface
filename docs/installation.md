# 🚀 Installation

## 📦 Install from PyPI

### CPU-only (default):

```bash
pip install uniface
```

This installs the CPU-compatible version of ONNX Runtime (`onnxruntime`) and all core dependencies.

### GPU support:

```bash
pip install "uniface[gpu]"
```

This installs `onnxruntime-gpu` for accelerated inference on supported NVIDIA GPUs.
Make sure your system meets the [ONNX Runtime GPU requirements](https://onnxruntime.ai/docs/build/eps.html#cuda).

---

## 🔧 Install from GitHub (latest version)

Clone the repository and install it manually:

```bash
git clone https://github.com/yakhyo/uniface.git
cd uniface

# CPU version
pip install .

# Or with GPU support
pip install ".[gpu]"
```
