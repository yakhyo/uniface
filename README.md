<h1 align="center">UniFace: All-in-One Face Analysis Library</h1>

<div align="center">

[![PyPI Version](https://img.shields.io/pypi/v/uniface.svg?label=Version)](https://pypi.org/project/uniface/)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Github Build Status](https://github.com/yakhyo/uniface/actions/workflows/ci.yml/badge.svg)](https://github.com/yakhyo/uniface/actions)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/uniface?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=Downloads)](https://pepy.tech/projects/uniface)
[![UniFace Documentation](https://img.shields.io/badge/Docs-UniFace-blue.svg)](https://yakhyo.github.io/uniface/)
[![Kaggle Badge](https://img.shields.io/badge/Notebooks-Kaggle?label=Kaggle&color=blue)](https://www.kaggle.com/yakhyokhuja/code)
[![Discord](https://img.shields.io/badge/Discord-Join%20Server-5865F2?logo=discord&logoColor=white)](https://discord.gg/wdzrjr7R5j)

</div>

<div align="center">
    <img src="https://raw.githubusercontent.com/yakhyo/uniface/main/.github/logos/uniface_rounded_q80.webp" width="90%" alt="UniFace - All-in-One Open-Source Face Analysis Library">
</div>

---

**UniFace** is a lightweight, production-ready face analysis library built on ONNX Runtime. It provides high-performance face detection, recognition, landmark detection, face parsing, gaze estimation, and attribute analysis with hardware acceleration support across platforms.

---

## Features

- **Face Detection** — RetinaFace, SCRFD, YOLOv5-Face, and YOLOv8-Face with 5-point landmarks
- **Face Recognition** — AdaFace, ArcFace, EdgeFace, MobileFace, and SphereFace embeddings
- **Face Tracking** — Multi-object tracking with [BYTETracker](https://github.com/yakhyo/bytetrack-tracker) for persistent IDs across video frames
- **Facial Landmarks** — 106-point landmark localization module (separate from 5-point detector landmarks)
- **Face Parsing** — BiSeNet semantic segmentation (19 classes), XSeg face masking
- **Portrait Matting** — Trimap-free alpha matte with MODNet (background removal, green screen, compositing)
- **Gaze Estimation** — Real-time gaze direction with MobileGaze
- **Head Pose Estimation** — 3D head orientation (pitch, yaw, roll) with 6D rotation representation
- **Attribute Analysis** — Age, gender, race (FairFace), and emotion
- **Vector Store** — FAISS-backed embedding store for fast multi-identity search
- **Anti-Spoofing** — Face liveness detection with MiniFASNet
- **Face Anonymization** — 5 blur methods for privacy protection
- **Hardware Acceleration** — ARM64 (Apple Silicon), CUDA (NVIDIA), CPU

---

## Visual Examples

<table>
  <tr>
    <td align="center"><b>Face Detection</b><br><img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demos/detection.jpg" width="100%"></td>
    <td align="center"><b>Gaze Estimation</b><br><img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demos/gaze.jpg" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><b>Head Pose Estimation</b><br><img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demos/headpose.jpg" width="100%"></td>
    <td align="center"><b>Age &amp; Gender</b><br><img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demos/age_gender.jpg" width="100%"></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><b>Face Verification</b><br><img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demos/verification.jpg" width="80%"></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><b>106-Point Landmarks</b><br><img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demos/landmarks.jpg" width="36%"></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><b>Face Parsing</b><br><img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demos/parsing.jpg" width="80%"></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><b>Face Segmentation</b><br><img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demos/segmentation.jpg" width="80%"></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><b>Portrait Matting</b><br><img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demos/matting.jpg" width="100%"></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><b>Face Anonymization</b><br><img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demos/anonymization.jpg" width="100%"></td>
  </tr>
</table>

---

## Installation

**Standard installation**

```bash
pip install uniface
```

**GPU support (CUDA)**

```bash
pip install uniface[gpu]
```

**From source (latest version)**

```bash
git clone https://github.com/yakhyo/uniface.git
cd uniface && pip install -e .
```

**FAISS vector store**

```bash
pip install faiss-cpu   # or faiss-gpu for CUDA
```

**Optional dependencies**
- Emotion model uses TorchScript and requires `torch`:
  `pip install torch` (choose the correct build for your OS/CUDA)
- YOLOv5-Face and YOLOv8-Face support faster NMS with `torchvision`:
  `pip install torch torchvision` then use `nms_mode='torchvision'`

---

## Model Downloads and Cache

Models are downloaded automatically on first use and verified via SHA-256.

Default cache location: `~/.uniface/models`

Override with the programmatic API or environment variable:

```python
from uniface.model_store import get_cache_dir, set_cache_dir

set_cache_dir('/data/models')
print(get_cache_dir())  # /data/models
```

```bash
export UNIFACE_CACHE_DIR=/data/models
```

---

## Quick Example (Detection)

```python
import cv2
from uniface.detection import RetinaFace

detector = RetinaFace()

image = cv2.imread("photo.jpg")
if image is None:
    raise ValueError("Failed to load image. Check the path to 'photo.jpg'.")

faces = detector.detect(image)

for face in faces:
    print(f"Confidence: {face.confidence:.2f}")
    print(f"BBox: {face.bbox}")
    print(f"Landmarks: {face.landmarks.shape}")
```

<div align="center">
    <img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/test_result.png" width="90%">
    <p>Face Detection Model Output</p>
</div>

---

## Example (Face Analyzer)

```python
import cv2
from uniface import FaceAnalyzer

# Zero-config: uses SCRFD (500M) + ArcFace (MobileNet) by default
analyzer = FaceAnalyzer()

image = cv2.imread("photo.jpg")
if image is None:
    raise ValueError("Failed to load image. Check the path to 'photo.jpg'.")

faces = analyzer.analyze(image)

for face in faces:
    print(face.bbox, face.embedding.shape if face.embedding is not None else None)
```

With attributes:

```python
from uniface import FaceAnalyzer, AgeGender

analyzer = FaceAnalyzer(attributes=[AgeGender()])
faces = analyzer.analyze(image)

for face in faces:
    print(f"{face.sex}, {face.age}y, embedding={face.embedding.shape}")
```

---

## Example (Portrait Matting)

```python
import cv2
import numpy as np
from uniface.matting import MODNet

matting = MODNet()

image = cv2.imread("portrait.jpg")
matte = matting.predict(image)  # (H, W) float32 in [0, 1]

# Transparent PNG
rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
rgba[:, :, 3] = (matte * 255).astype(np.uint8)
cv2.imwrite("transparent.png", rgba)

# Green screen
matte_3ch = matte[:, :, np.newaxis]
bg = np.full_like(image, (0, 177, 64), dtype=np.uint8)
result = (image * matte_3ch + bg * (1 - matte_3ch)).astype(np.uint8)
cv2.imwrite("green_screen.jpg", result)
```

---

## Jupyter Notebooks

| Example | Colab | Description |
|---------|:-----:|-------------|
| [01_face_detection.ipynb](examples/01_face_detection.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yakhyo/uniface/blob/main/examples/01_face_detection.ipynb) | Face detection and landmarks |
| [02_face_alignment.ipynb](examples/02_face_alignment.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yakhyo/uniface/blob/main/examples/02_face_alignment.ipynb) | Face alignment for recognition |
| [03_face_verification.ipynb](examples/03_face_verification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yakhyo/uniface/blob/main/examples/03_face_verification.ipynb) | Compare faces for identity |
| [04_face_search.ipynb](examples/04_face_search.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yakhyo/uniface/blob/main/examples/04_face_search.ipynb) | Find a person in group photos |
| [05_face_analyzer.ipynb](examples/05_face_analyzer.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yakhyo/uniface/blob/main/examples/05_face_analyzer.ipynb) | All-in-one analysis |
| [06_face_parsing.ipynb](examples/06_face_parsing.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yakhyo/uniface/blob/main/examples/06_face_parsing.ipynb) | Semantic face segmentation |
| [07_face_anonymization.ipynb](examples/07_face_anonymization.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yakhyo/uniface/blob/main/examples/07_face_anonymization.ipynb) | Privacy-preserving blur |
| [08_gaze_estimation.ipynb](examples/08_gaze_estimation.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yakhyo/uniface/blob/main/examples/08_gaze_estimation.ipynb) | Gaze direction estimation |
| [09_face_segmentation.ipynb](examples/09_face_segmentation.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yakhyo/uniface/blob/main/examples/09_face_segmentation.ipynb) | Face segmentation with XSeg |
| [10_face_vector_store.ipynb](examples/10_face_vector_store.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yakhyo/uniface/blob/main/examples/10_face_vector_store.ipynb) | FAISS-backed face database |
| [11_head_pose_estimation.ipynb](examples/11_head_pose_estimation.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yakhyo/uniface/blob/main/examples/11_head_pose_estimation.ipynb) | Head pose estimation (pitch, yaw, roll) |
| [12_face_recognition.ipynb](examples/12_face_recognition.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yakhyo/uniface/blob/main/examples/12_face_recognition.ipynb) | Standalone face recognition pipeline |
| [13_portrait_matting.ipynb](examples/13_portrait_matting.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yakhyo/uniface/blob/main/examples/13_portrait_matting.ipynb) | Portrait matting with MODNet |

---

## Documentation

Full documentation: https://yakhyo.github.io/uniface/

| Resource | Description |
|----------|-------------|
| [Quickstart](https://yakhyo.github.io/uniface/quickstart/) | Get up and running in 5 minutes |
| [Model Zoo](https://yakhyo.github.io/uniface/models/) | All models, benchmarks, and selection guide |
| [API Reference](https://yakhyo.github.io/uniface/modules/detection/) | Detailed module documentation |
| [Tutorials](https://yakhyo.github.io/uniface/recipes/image-pipeline/) | Step-by-step workflow examples |
| [Guides](https://yakhyo.github.io/uniface/concepts/overview/) | Architecture and design principles |
| [Datasets](https://yakhyo.github.io/uniface/datasets/) | Training data and evaluation benchmarks |

---

## Execution Providers (ONNX Runtime)

```python
from uniface.detection import RetinaFace

# Force CPU-only inference
detector = RetinaFace(providers=["CPUExecutionProvider"])
```

See more in the docs:
https://yakhyo.github.io/uniface/concepts/execution-providers/

---

## Datasets

| Task | Training Dataset | Models |
|------|-----------------|--------|
| Detection | WIDER FACE | RetinaFace, SCRFD, YOLOv5-Face, YOLOv8-Face |
| Recognition | MS1MV2 | MobileFace, SphereFace |
| Recognition | WebFace600K | ArcFace |
| Recognition | WebFace4M / 12M | AdaFace |
| Recognition | MS1MV2 | EdgeFace |
| Gaze | Gaze360 | MobileGaze |
| Head Pose | 300W-LP | HeadPose (ResNet, MobileNet) |
| Parsing | CelebAMask-HQ | BiSeNet |
| Attributes | CelebA, FairFace, AffectNet | AgeGender, FairFace, Emotion |

> See [Datasets documentation](https://yakhyo.github.io/uniface/datasets/) for download links, benchmarks, and details.

---

## Licensing and Model Usage

UniFace is MIT-licensed, but several pretrained models carry their own licenses.
Review: https://yakhyo.github.io/uniface/license-attribution/

Notable examples:
- YOLOv5-Face and YOLOv8-Face weights are GPL-3.0
- FairFace weights are CC BY 4.0

If you plan commercial use, verify model license compatibility.

---

## References

| Feature | Repository | Training | Description |
|---------|------------|:--------:|-------------|
| Detection | [retinaface-pytorch](https://github.com/yakhyo/retinaface-pytorch) | ✓ | RetinaFace PyTorch Training & Export |
| Detection | [yolov5-face-onnx-inference](https://github.com/yakhyo/yolov5-face-onnx-inference) | - | YOLOv5-Face ONNX Inference |
| Detection | [yolov8-face-onnx-inference](https://github.com/yakhyo/yolov8-face-onnx-inference) | - | YOLOv8-Face ONNX Inference |
| Tracking | [bytetrack-tracker](https://github.com/yakhyo/bytetrack-tracker) | - | BYTETracker Multi-Object Tracking |
| Recognition | [face-recognition](https://github.com/yakhyo/face-recognition) | ✓ | MobileFace, SphereFace Training |
| Recognition | [edgeface-onnx](https://github.com/yakhyo/edgeface-onnx) | - | EdgeFace ONNX Inference |
| Parsing | [face-parsing](https://github.com/yakhyo/face-parsing) | ✓ | BiSeNet Face Parsing |
| Parsing | [face-segmentation](https://github.com/yakhyo/face-segmentation) | - | XSeg Face Segmentation |
| Gaze | [gaze-estimation](https://github.com/yakhyo/gaze-estimation) | ✓ | MobileGaze Training |
| Head Pose | [head-pose-estimation](https://github.com/yakhyo/head-pose-estimation) | ✓ | Head Pose Training (6DRepNet-style) |
| Matting | [modnet](https://github.com/yakhyo/modnet) | - | MODNet Portrait Matting |
| Anti-Spoofing | [face-anti-spoofing](https://github.com/yakhyo/face-anti-spoofing) | - | MiniFASNet Inference |
| Attributes | [fairface-onnx](https://github.com/yakhyo/fairface-onnx) | - | FairFace ONNX Inference |

*SCRFD and ArcFace models are from [InsightFace](https://github.com/deepinsight/insightface).

---

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md).

## Support

If you find this project useful, consider giving it a ⭐ on GitHub — it helps others discover it!

Questions or feedback:
- Discord: https://discord.gg/wdzrjr7R5j
- GitHub Issues: https://github.com/yakhyo/uniface/issues
- DeepWiki Q&A: https://deepwiki.com/yakhyo/uniface

## License

This project is licensed under the [MIT License](LICENSE).

> **Disclaimer:** This project is not affiliated with or related to
> [Uniface](https://uniface.com/) by Rocket Software.
