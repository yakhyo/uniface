# Troubleshooting

Common issues and solutions.

---

## Installation Issues

### Import Error

```
ModuleNotFoundError: No module named 'uniface'
```

**Solution:** Install the package:

```bash
pip install uniface
```

### Python Version

```
Python 3.10+ required
```

**Solution:** Check your Python version:

```bash
python --version  # Should be 3.11+
```

---

## Model Issues

### Model Download Failed

```
Failed to download model
```

**Solution:** Manually download:

```python
from uniface.model_store import verify_model_weights
from uniface.constants import RetinaFaceWeights

path = verify_model_weights(RetinaFaceWeights.MNET_V2)
```

### Model Not Found

**Solution:** Check cache directory:

```bash
ls ~/.uniface/models/
```

---

## Performance Issues

### Slow on Mac

**Check:** Verify ARM64 Python:

```bash
python -c "import platform; print(platform.machine())"
# Should show: arm64
```

### No GPU Acceleration

**Check:** Verify CUDA:

```python
import onnxruntime as ort
print(ort.get_available_providers())
# Should include 'CUDAExecutionProvider'
```

**Solution:** Install GPU version:

```bash
pip install uniface[gpu]
```

---

## Detection Issues

### No Faces Detected

**Try:**

1. Lower confidence threshold:
   ```python
   detector = RetinaFace(confidence_threshold=0.3)
   ```

2. Check image format (should be BGR):
   ```python
   image = cv2.imread("photo.jpg")  # BGR format
   ```

### Wrong Bounding Boxes

**Check:** Image orientation. Some cameras return rotated images.

---

## Recognition Issues

### Low Similarity Scores

**Try:**

1. Ensure face alignment is working
2. Use higher quality images
3. Check lighting conditions

### Different Results Each Time

**Note:** Results should be deterministic. If not, check:

- Image preprocessing
- Model loading

---

## Memory Issues

### Out of Memory

**Solutions:**

1. Process images in batches
2. Use smaller input size:
   ```python
   detector = RetinaFace(input_size=(320, 320))
   ```
3. Release resources:
   ```python
   del detector
   import gc
   gc.collect()
   ```

---

## Still Having Issues?

1. Check [GitHub Issues](https://github.com/yakhyo/uniface/issues)
2. Open a new issue with:
   - Python version
   - UniFace version
   - Error message
   - Minimal code to reproduce
