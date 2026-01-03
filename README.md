# UniFace: All-in-One Face Analysis Library

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/uniface.svg)](https://pypi.org/project/uniface/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/yakhyo/uniface/actions/workflows/ci.yml/badge.svg)](https://github.com/yakhyo/uniface/actions)
[![Downloads](https://static.pepy.tech/badge/uniface)](https://pepy.tech/project/uniface)
[![Docs](https://img.shields.io/badge/Docs-UniFace-blue.svg)](https://yakhyo.github.io/uniface/)

</div>

<div align="center">
    <img src=".github/logos/logo_web.webp" width=80%>
</div>

**UniFace** is a lightweight, production-ready face analysis library built on ONNX Runtime. It provides high-performance face detection, recognition, landmark detection, face parsing, gaze estimation, and attribute analysis with hardware acceleration support across platforms.

> 💬 **Have questions?** [Chat with this codebase on DeepWiki](https://deepwiki.com/yakhyo/uniface) - AI-powered docs that let you ask anything about UniFace.

---

## Features

- **Face Detection** — RetinaFace, SCRFD, YOLOv5-Face, and YOLOv8-Face with 5-point landmarks
- **Face Recognition** — ArcFace, MobileFace, and SphereFace embeddings
- **Facial Landmarks** — 106-point landmark localization
- **Face Parsing** — BiSeNet semantic segmentation (19 classes)
- **Gaze Estimation** — Real-time gaze direction with MobileGaze
- **Attribute Analysis** — Age, gender, race (FairFace), and emotion
- **Anti-Spoofing** — Face liveness detection with MiniFASNet
- **Face Anonymization** — 5 blur methods for privacy protection
- **Hardware Acceleration** — ARM64 (Apple Silicon), CUDA (NVIDIA), CPU

---

## Installation

```bash
# Standard installation
pip install uniface

# GPU support (CUDA)
pip install uniface[gpu]

# From source
git clone https://github.com/yakhyo/uniface.git
cd uniface && pip install -e .
```

---

## Quick Example

```python
import cv2
from uniface import RetinaFace

# Initialize detector (models auto-download on first use)
detector = RetinaFace()

# Detect faces
image = cv2.imread("photo.jpg")
faces = detector.detect(image)

for face in faces:
    print(f"Confidence: {face.confidence:.2f}")
    print(f"BBox: {face.bbox}")
    print(f"Landmarks: {face.landmarks.shape}")
```

<div align="center">
    <img src="assets/test_result.png">
</div>

---

## Documentation

📚 **Full documentation**: [yakhyo.github.io/uniface](https://yakhyo.github.io/uniface/)

| Resource | Description |
|----------|-------------|
| [Quickstart](https://yakhyo.github.io/uniface/quickstart/) | Get up and running in 5 minutes |
| [Model Zoo](https://yakhyo.github.io/uniface/models/) | All models, benchmarks, and selection guide |
| [API Reference](https://yakhyo.github.io/uniface/modules/detection/) | Detailed module documentation |
| [Tutorials](https://yakhyo.github.io/uniface/recipes/image-pipeline/) | Step-by-step workflow examples |
| [Guides](https://yakhyo.github.io/uniface/concepts/overview/) | Architecture and design principles |

### Jupyter Notebooks

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

---

## References

- [yakhyo/retinaface-pytorch](https://github.com/yakhyo/retinaface-pytorch) — RetinaFace training
- [yakhyo/yolov5-face-onnx-inference](https://github.com/yakhyo/yolov5-face-onnx-inference) — YOLOv5-Face ONNX
- [yakhyo/yolov8-face-onnx-inference](https://github.com/yakhyo/yolov8-face-onnx-inference) — YOLOv8-Face ONNX
- [yakhyo/face-recognition](https://github.com/yakhyo/face-recognition) — ArcFace, MobileFace, SphereFace
- [yakhyo/face-parsing](https://github.com/yakhyo/face-parsing) — BiSeNet face parsing
- [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation) — MobileGaze training
- [yakhyo/face-anti-spoofing](https://github.com/yakhyo/face-anti-spoofing) — MiniFASNet inference
- [yakhyo/fairface-onnx](https://github.com/yakhyo/fairface-onnx) — FairFace attributes
- [deepinsight/insightface](https://github.com/deepinsight/insightface) — Model architectures

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the [MIT License](LICENSE).
