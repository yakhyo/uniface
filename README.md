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
    <img src="https://raw.githubusercontent.com/yakhyo/uniface/main/.github/logos/new/uniface_rounded_q80.webp" width="90%" alt="UniFace - All-in-One Open-Source Face Analysis Library">
</div>

---

**UniFace** is a lightweight, production-ready face analysis library built on ONNX Runtime. It provides high-performance face detection, recognition, landmark detection, face parsing, gaze estimation, and attribute analysis with hardware acceleration support across platforms.

---

## Features

- **Face Detection** — RetinaFace, SCRFD, YOLOv5-Face, and YOLOv8-Face with 5-point landmarks
- **Face Recognition** — ArcFace, MobileFace, and SphereFace embeddings
- **Face Tracking** — Multi-object tracking with [BYTETracker](https://github.com/yakhyo/bytetrack-tracker) for persistent IDs across video frames
- **Facial Landmarks** — 106-point landmark localization module (separate from 5-point detector landmarks)
- **Face Parsing** — BiSeNet semantic segmentation (19 classes), XSeg face masking
- **Gaze Estimation** — Real-time gaze direction with MobileGaze
- **Attribute Analysis** — Age, gender, race (FairFace), and emotion
- **Anti-Spoofing** — Face liveness detection with MiniFASNet
- **Face Anonymization** — 5 blur methods for privacy protection
- **Hardware Acceleration** — ARM64 (Apple Silicon), CUDA (NVIDIA), CPU

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
from uniface.analyzer import FaceAnalyzer
from uniface.detection import RetinaFace
from uniface.recognition import ArcFace

detector = RetinaFace()
recognizer = ArcFace()

analyzer = FaceAnalyzer(detector, recognizer=recognizer)

image = cv2.imread("photo.jpg")
if image is None:
    raise ValueError("Failed to load image. Check the path to 'photo.jpg'.")

faces = analyzer.analyze(image)

for face in faces:
    print(face.bbox, face.embedding.shape if face.embedding is not None else None)
```

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

## Documentation

Full documentation: https://yakhyo.github.io/uniface/

| Resource | Description |
|----------|-------------|
| [Quickstart](https://yakhyo.github.io/uniface/quickstart/) | Get up and running in 5 minutes |
| [Model Zoo](https://yakhyo.github.io/uniface/models/) | All models, benchmarks, and selection guide |
| [API Reference](https://yakhyo.github.io/uniface/modules/detection/) | Detailed module documentation |
| [Tutorials](https://yakhyo.github.io/uniface/recipes/image-pipeline/) | Step-by-step workflow examples |
| [Guides](https://yakhyo.github.io/uniface/concepts/overview/) | Architecture and design principles |

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
| Parsing | [face-parsing](https://github.com/yakhyo/face-parsing) | ✓ | BiSeNet Face Parsing |
| Parsing | [face-segmentation](https://github.com/yakhyo/face-segmentation) | - | XSeg Face Segmentation |
| Gaze | [gaze-estimation](https://github.com/yakhyo/gaze-estimation) | ✓ | MobileGaze Training |
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
