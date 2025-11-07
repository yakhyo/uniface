# Testing Scripts Guide

Complete guide to testing all scripts in the `scripts/` directory.

---

## 📁 Available Scripts

1. **download_model.py** - Download and verify model weights
2. **run_detection.py** - Face detection on images
3. **run_recognition.py** - Face recognition (extract embeddings)
4. **run_face_search.py** - Real-time face matching with webcam
5. **sha256_generate.py** - Generate SHA256 checksums for models

---

## Testing Each Script

### 1. Test Model Download

```bash
# Download a specific model
python scripts/download_model.py --model MNET_V2

# Download all RetinaFace models (takes ~5 minutes, ~200MB)
python scripts/download_model.py

# Verify models are cached
ls -lh ~/.uniface/models/
```

**Expected Output:**
```
📥 Downloading model: retinaface_mnet_v2
2025-11-08 00:00:00 - INFO - Downloading model 'RetinaFaceWeights.MNET_V2' from https://...
Downloading ~/.uniface/models/retinaface_mnet_v2.onnx: 100%|████| 3.5M/3.5M
2025-11-08 00:00:05 - INFO - Successfully downloaded 'RetinaFaceWeights.MNET_V2'
✅ All requested weights are ready and verified.
```

---

### 2. Test Face Detection

```bash
# Basic detection
python scripts/run_detection.py --image assets/test.jpg

# With custom settings
python scripts/run_detection.py \
    --image assets/test.jpg \
    --method scrfd \
    --threshold 0.7 \
    --save_dir outputs

# Benchmark mode (100 iterations)
python scripts/run_detection.py \
    --image assets/test.jpg \
    --iterations 100
```

**Expected Output:**
```
Initializing detector: retinaface
2025-11-08 00:00:00 - INFO - Initializing RetinaFace with model=RetinaFaceWeights.MNET_V2...
2025-11-08 00:00:01 - INFO - CoreML acceleration enabled (Apple Silicon)
✅ Output saved at: outputs/test_out.jpg
[1/1] ⏱️ Inference time: 0.0234 seconds
```

**Verify Output:**
```bash
# Check output image was created
ls -lh outputs/test_out.jpg

# View the image (macOS)
open outputs/test_out.jpg
```

---

### 3. Test Face Recognition (Embedding Extraction)

```bash
# Extract embeddings from an image
python scripts/run_recognition.py --image assets/test.jpg

# With different models
python scripts/run_recognition.py \
    --image assets/test.jpg \
    --detector scrfd \
    --recognizer mobileface
```

**Expected Output:**
```
Initializing detector: retinaface
Initializing recognizer: arcface
2025-11-08 00:00:00 - INFO - Successfully initialized face encoder from ~/.uniface/models/w600k_mbf.onnx
Detected 1 face(s). Extracting embeddings for the first face...
  - Embedding shape: (1, 512)
  - L2 norm of unnormalized embedding: 64.2341
  - L2 norm of normalized embedding: 1.0000
```

---

### 4. Test Real-Time Face Search (Webcam)

**Prerequisites:**
- Webcam connected
- Reference image with a clear face

```bash
# Basic usage
python scripts/run_face_search.py --image assets/test.jpg

# With custom models
python scripts/run_face_search.py \
    --image assets/test.jpg \
    --detector scrfd \
    --recognizer arcface
```

**Expected Behavior:**
1. Webcam window opens
2. Faces are detected in real-time
3. Green box = Match (similarity > 0.4)
4. Red box = Unknown (similarity < 0.4)
5. Press 'q' to quit

**Expected Output:**
```
Initializing models...
2025-11-08 00:00:00 - INFO - CoreML acceleration enabled (Apple Silicon)
Extracting reference embedding...
Webcam started. Press 'q' to quit.
```

**Troubleshooting:**
```bash
# If webcam doesn't open
python -c "import cv2; cap = cv2.VideoCapture(0); print('Webcam OK' if cap.isOpened() else 'Webcam FAIL')"

# If no faces detected
# - Ensure good lighting
# - Face should be frontal and clearly visible
# - Try lowering threshold: edit script line 29, change 0.4 to 0.3
```

---

### 5. Test SHA256 Generator (For Developers)

```bash
# Generate checksum for a model file
python scripts/sha256_generate.py ~/.uniface/models/retinaface_mnet_v2.onnx

# Generate for all models
for model in ~/.uniface/models/*.onnx; do
    python scripts/sha256_generate.py "$model"
done
```

---

## 🔍 Quick Verification Tests

### Test 1: Imports Work

```bash
python -c "
from uniface.detection import create_detector
from uniface.recognition import create_recognizer
print('✅ Imports successful')
"
```

### Test 2: Models Download

```bash
python -c "
from uniface import RetinaFace
detector = RetinaFace()
print('✅ Model downloaded and loaded')
"
```

### Test 3: Detection Works

```bash
python -c "
import cv2
import numpy as np
from uniface import RetinaFace

detector = RetinaFace()
image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
faces = detector.detect(image)
print(f'✅ Detection works, found {len(faces)} faces')
"
```

### Test 4: Recognition Works

```bash
python -c "
import cv2
import numpy as np
from uniface import RetinaFace, ArcFace

detector = RetinaFace()
recognizer = ArcFace()
image = cv2.imread('assets/test.jpg')
faces = detector.detect(image)
if faces:
    landmarks = np.array(faces[0]['landmarks'])
    embedding = recognizer.get_normalized_embedding(image, landmarks)
    print(f'✅ Recognition works, embedding shape: {embedding.shape}')
else:
    print('⚠️ No faces detected in test image')
"
```

---

## End-to-End Test Workflow

Run this complete workflow to verify everything works:

```bash
#!/bin/bash
# Save as test_all_scripts.sh

echo "=== Testing UniFace Scripts ==="
echo ""

# Test 1: Download models
echo "1️⃣ Testing model download..."
python scripts/download_model.py --model MNET_V2
if [ $? -eq 0 ]; then
    echo "✅ Model download: PASS"
else
    echo "❌ Model download: FAIL"
    exit 1
fi
echo ""

# Test 2: Face detection
echo "2️⃣ Testing face detection..."
python scripts/run_detection.py --image assets/test.jpg --save_dir /tmp/uniface_test
if [ $? -eq 0 ] && [ -f /tmp/uniface_test/test_out.jpg ]; then
    echo "✅ Face detection: PASS"
else
    echo "❌ Face detection: FAIL"
    exit 1
fi
echo ""

# Test 3: Face recognition
echo "3️⃣ Testing face recognition..."
python scripts/run_recognition.py --image assets/test.jpg > /tmp/uniface_recognition.log
if [ $? -eq 0 ] && grep -q "Embedding shape" /tmp/uniface_recognition.log; then
    echo "✅ Face recognition: PASS"
else
    echo "❌ Face recognition: FAIL"
    exit 1
fi
echo ""

echo "=== All Tests Passed! 🎉 ==="
```

**Run the test suite:**
```bash
chmod +x test_all_scripts.sh
./test_all_scripts.sh
```

---

## Performance Benchmarking

### Benchmark Detection Speed

```bash
# Test different models
for model in retinaface scrfd; do
    echo "Testing $model..."
    python scripts/run_detection.py \
        --image assets/test.jpg \
        --method $model \
        --iterations 50
done
```

### Benchmark Recognition Speed

```bash
# Test different recognizers
for recognizer in arcface mobileface; do
    echo "Testing $recognizer..."
    time python scripts/run_recognition.py \
        --image assets/test.jpg \
        --recognizer $recognizer
done
```

---

## 🐛 Common Issues

### Issue: "No module named 'uniface'"

```bash
# Solution: Install in editable mode
pip install -e .
```

### Issue: "Failed to load image"

```bash
# Check image exists
ls -lh assets/test.jpg

# Try with absolute path
python scripts/run_detection.py --image $(pwd)/assets/test.jpg
```

### Issue: "No faces detected"

```bash
# Lower confidence threshold
python scripts/run_detection.py \
    --image assets/test.jpg \
    --threshold 0.3
```

### Issue: Models downloading slowly

```bash
# Check internet connection
curl -I https://github.com/yakhyo/uniface/releases

# Or download manually
wget https://github.com/yakhyo/uniface/releases/download/v0.1.2/retinaface_mv2.onnx \
    -O ~/.uniface/models/retinaface_mnet_v2.onnx
```

### Issue: CoreML not available on Mac

```bash
# Install CoreML-enabled ONNX Runtime
pip uninstall onnxruntime
pip install onnxruntime-silicon

# Verify
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Should show: ['CoreMLExecutionProvider', 'CPUExecutionProvider']
```

---

## ✅ Script Status Summary

| Script                | Status | API Updated | Tested |
|-----------------------|--------|-------------|--------|
| download_model.py     | ✅     | ✅          | ✅     |
| run_detection.py      | ✅     | ✅          | ✅     |
| run_recognition.py    | ✅     | ✅          | ✅     |
| run_face_search.py    | ✅     | ✅          | ✅     |
| sha256_generate.py    | ✅     | N/A         | ✅     |

All scripts are updated and working with the new dict-based API! 🎉

---

## 📝 Notes

- All scripts now use the factory functions (`create_detector`, `create_recognizer`)
- Scripts work with the new dict-based detection API
- Model download bug is fixed (enum vs string issue)
- CoreML acceleration is automatically detected on Apple Silicon
- All scripts include proper error handling

---

Need help with a specific script? Check the main [README.md](../README.md) or [QUICKSTART.md](../QUICKSTART.md)!

