# Execution Providers

UniFace uses ONNX Runtime for model inference, which supports multiple hardware acceleration backends.

---

## Automatic Provider Selection

UniFace automatically selects the optimal execution provider based on available hardware:

```python
from uniface.detection import RetinaFace

# Automatically uses best available provider
detector = RetinaFace()
```

**Priority order:**

1. **CoreMLExecutionProvider** - Apple Silicon
2. **CUDAExecutionProvider** - NVIDIA GPU
3. **CPUExecutionProvider** - Fallback

---

## Explicit Provider Selection

You can specify which execution provider to use by passing the `providers` parameter:

```python
from uniface.detection import RetinaFace
from uniface.recognition import ArcFace

# Force CPU execution (even if GPU is available)
detector = RetinaFace(providers=['CPUExecutionProvider'])
recognizer = ArcFace(providers=['CPUExecutionProvider'])

# Use CUDA with CPU fallback
detector = RetinaFace(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
```

All model classes accept the `providers` parameter:

- Detection: `RetinaFace`, `SCRFD`, `YOLOv5Face`, `YOLOv8Face`
- Recognition: `ArcFace`, `AdaFace`, `MobileFace`, `SphereFace`
- Landmarks: `Landmark106`
- Gaze: `MobileGaze`
- Parsing: `BiSeNet`
- Attributes: `AgeGender`, `FairFace`
- Anti-Spoofing: `MiniFASNet`

---

## Check Available Providers

```python
import onnxruntime as ort

providers = ort.get_available_providers()
print("Available providers:", providers)
```

**Example outputs:**

=== "macOS (Apple Silicon)"

    ```
    ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    ```

=== "Linux (NVIDIA GPU)"

    ```
    ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ```

=== "Windows (CPU)"

    ```
    ['CPUExecutionProvider']
    ```

---

## Platform-Specific Setup

### Apple Silicon (M1/M2/M3/M4)

No additional setup required. ARM64 optimizations are built into `onnxruntime`:

```bash
pip install uniface
```

Verify ARM64:

```bash
python -c "import platform; print(platform.machine())"
# Should show: arm64
```

!!! tip "Performance"
    Apple Silicon Macs use CoreML acceleration automatically, providing excellent performance for face analysis tasks.

---

### NVIDIA GPU (CUDA)

Install with GPU support:

```bash
pip install uniface[gpu]
```

**Requirements:**

- CUDA 11.x or 12.x
- cuDNN 8.x
- Compatible NVIDIA driver

Verify CUDA:

```python
import onnxruntime as ort

if 'CUDAExecutionProvider' in ort.get_available_providers():
    print("CUDA is available!")
else:
    print("CUDA not available, using CPU")
```

---

### CPU Fallback

CPU execution is always available:

```bash
pip install uniface
```

Works on all platforms without additional configuration.

---

## Internal API

For advanced use cases, you can access the provider utilities:

```python
from uniface.onnx_utils import get_available_providers, create_onnx_session

# Check available providers
providers = get_available_providers()
print(f"Available: {providers}")

# Models use create_onnx_session() internally
# which auto-selects the best provider
```

---

## Performance Tips

### 1. Use GPU When Available

For batch processing or real-time applications, GPU acceleration provides significant speedups:

```bash
pip install uniface[gpu]
```

### 2. Optimize Input Size

Smaller input sizes are faster but may reduce accuracy:

```python
from uniface.detection import RetinaFace

# Faster, lower accuracy
detector = RetinaFace(input_size=(320, 320))

# Balanced (default)
detector = RetinaFace(input_size=(640, 640))
```

### 3. Batch Processing

Process multiple images to maximize GPU utilization:

```python
# Process images in batch (GPU-efficient)
for image_path in image_paths:
    image = cv2.imread(image_path)
    faces = detector.detect(image)
    # ...
```

---

## Troubleshooting

### CUDA Not Detected

1. Verify CUDA installation:
   ```bash
   nvidia-smi
   ```

2. Check CUDA version compatibility with ONNX Runtime

3. Reinstall with GPU support:
   ```bash
   pip uninstall onnxruntime onnxruntime-gpu
   pip install uniface[gpu]
   ```

### Slow Performance on Mac

Verify you're using ARM64 Python (not Rosetta):

```bash
python -c "import platform; print(platform.machine())"
# Should show: arm64 (not x86_64)
```

---

## Next Steps

- [Model Cache & Offline](model-cache-offline.md) - Model management
- [Thresholds & Calibration](thresholds-calibration.md) - Tuning parameters
