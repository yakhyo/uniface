<h1 align="center">UniFace: A Unified Face Analysis Library for Python</h1>

<div align="center">

[![PyPI Version](https://img.shields.io/pypi/v/uniface.svg?label=Version)](https://pypi.org/project/uniface/)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Github Build Status](https://github.com/yakhyo/uniface/actions/workflows/ci.yml/badge.svg)](https://github.com/yakhyo/uniface/actions)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/uniface?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=Downloads)](https://pepy.tech/projects/uniface)
[![Kaggle Badge](https://img.shields.io/badge/Notebooks-Kaggle?label=Kaggle&color=blue)](https://www.kaggle.com/yakhyokhuja/code)

</div>

<div align="center">
    <img src="https://raw.githubusercontent.com/yakhyo/uniface/main/.github/logos/uniface_rounded_q80.webp" width="90%" alt="UniFace - A Unified Face Analysis Library for Python">
</div>

<p align="center">
  UniFace is a lightweight, production-ready Python library for face detection, recognition,<br>
  tracking, landmark analysis, face parsing, gaze estimation, and face attributes.
</p>

<p align="center">
  <a href="https://yakhyo.github.io/uniface/quickstart/"><img src="https://img.shields.io/badge/Get%20Started-1f6feb?style=for-the-badge&logoColor=white" alt="Get Started"></a>
  &nbsp;
  <a href="https://yakhyo.github.io/uniface/models/"><img src="https://img.shields.io/badge/Model%20Zoo-30363d?style=for-the-badge&logoColor=white" alt="Model Zoo"></a>
  &nbsp;
  <a href="https://yakhyo.github.io/uniface/notebooks/"><img src="https://img.shields.io/badge/Notebooks-30363d?style=for-the-badge&logo=jupyter&logoColor=white" alt="Notebooks"></a>
  &nbsp;
  <a href="https://yakhyo.github.io/uniface/"><img src="https://img.shields.io/badge/Full%20Docs-30363d?style=for-the-badge&logoColor=white" alt="Full Docs"></a>
</p>

```bash
pip install "uniface[cpu]"          # CPU and Apple Silicon
pip install "uniface[gpu]"          # NVIDIA CUDA
pip install --pre "uniface[cpu]"    # latest pre-release
```

<details>
<summary><b>A first script</b></summary>

<br>

`FaceAnalyzer` runs detection, alignment and recognition in one call. Attribute models are opt-in.

```python
import cv2
from uniface import FaceAnalyzer, FairFace

analyzer = FaceAnalyzer(predictors=[FairFace()])

for face in analyzer.analyze(cv2.imread("photo.jpg")):
    print(face.bbox, face.sex, face.age_group, face.embedding.shape)
```

`bbox`, `confidence`, `landmarks` and `embedding` are always set. Age, sex, race, emotion, quality
and the face states stay `None` until you pass the predictor that fills them.

</details>

<details>
<summary><b>All fifteen tasks, and which model does each</b></summary>

<br>

| Task | Models |
| --- | --- |
| Face Detection | RetinaFace, SCRFD, CenterFace, YOLOv5-Face, YOLOv8-Face, BlazeFace |
| Face Recognition | AdaFace, ArcFace, EdgeFace, MobileFace, SphereFace |
| Face Tracking | BYTETracker, persistent IDs across video frames |
| Facial Landmarks | 2d106det (106), PIPNet (98 / 68), Face Mesh (468 / 478, 3D) |
| Face Parsing | BiSeNet (19 classes), XSeg masking |
| Portrait Matting | MODNet, trimap-free |
| Gaze Estimation | MobileGaze (ResNet-18 / 34 / 50, MobileNetV2) |
| Head Pose | 6D rotation representation, pitch / yaw / roll |
| Demographics | AgeGender, FairFace (age group, sex, race) |
| Emotion | AffectNet-7 and AffectNet-8 |
| Face States | FaceAttribNet: eyes, glasses, sunglasses, mask |
| Face Quality | eDifFIQA (T / S / M / L) |
| Anti-Spoofing | MiniFASNet liveness |
| Anonymization | 5 blur methods |
| Vector Store | FAISS-backed embedding search |

Runs on CPU, Apple Silicon and CUDA. Weights download on first use, verified by SHA-256.

</details>

<br>

### Find and measure faces

**Face Detection** &nbsp;·&nbsp; [docs](https://yakhyo.github.io/uniface/modules/detection/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/detection.jpg" width="100%">

**Facial Landmarks** &nbsp;·&nbsp; [docs](https://yakhyo.github.io/uniface/modules/landmarks/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/landmarks.jpg" width="100%">

**Face Mesh** &nbsp;·&nbsp; [docs](https://yakhyo.github.io/uniface/modules/landmarks/#face-mesh-468-or-478-points-3d)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/face_mesh.jpg" width="100%">

**Face Quality** &nbsp;·&nbsp; [docs](https://yakhyo.github.io/uniface/modules/quality/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/quality.jpg" width="100%">

### Cut faces out

**Face Parsing** &nbsp;·&nbsp; [docs](https://yakhyo.github.io/uniface/modules/parsing/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/parsing.jpg" width="100%">

**Face Segmentation** &nbsp;·&nbsp; [docs](https://yakhyo.github.io/uniface/modules/parsing/#xseg)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/segmentation.jpg" width="100%">

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/segmentation_occluded.jpg" width="100%">

**Portrait Matting** &nbsp;·&nbsp; [docs](https://yakhyo.github.io/uniface/modules/matting/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/matting.jpg" width="100%">

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/matting_alt.jpg" width="100%">

### Read where a head is pointing

**Head Pose** &nbsp;·&nbsp; [docs](https://yakhyo.github.io/uniface/modules/headpose/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/headpose.jpg" width="100%">

**Gaze Estimation** &nbsp;·&nbsp; [docs](https://yakhyo.github.io/uniface/modules/gaze/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/gaze.jpg" width="100%">

### Read a face

**Age and Sex** &nbsp;·&nbsp; [docs](https://yakhyo.github.io/uniface/modules/attributes/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/demography.jpg" width="100%">

**Emotion** &nbsp;·&nbsp; [docs](https://yakhyo.github.io/uniface/modules/attributes/#emotion)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/emotion.jpg" width="100%">

**Face States** &nbsp;·&nbsp; [docs](https://yakhyo.github.io/uniface/modules/attributes/#faceattribnet)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/face_states.jpg" width="100%">

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/face_states_alt.jpg" width="100%">

### Tell a real face from a replay

**Anti-Spoofing** &nbsp;·&nbsp; [docs](https://yakhyo.github.io/uniface/modules/spoofing/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/spoofing.jpg" width="100%">

### Match a face, or hide one

**Face Recognition** &nbsp;·&nbsp; [docs](https://yakhyo.github.io/uniface/modules/recognition/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/verification.jpg" width="100%">

**Face Anonymization** &nbsp;·&nbsp; [docs](https://yakhyo.github.io/uniface/modules/privacy/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/anonymization.jpg" width="100%">

<br>

<div align="center">

**[Get Started](https://yakhyo.github.io/uniface/quickstart/)** &nbsp;·&nbsp;
[Model Zoo](https://yakhyo.github.io/uniface/models/) &nbsp;·&nbsp;
[Notebooks](https://yakhyo.github.io/uniface/notebooks/) &nbsp;·&nbsp;
[Model licences](https://yakhyo.github.io/uniface/license-attribution/) &nbsp;·&nbsp;
[Contributing](CONTRIBUTING.md) &nbsp;·&nbsp;
[Discord](https://discord.gg/wdzrjr7R5j) &nbsp;·&nbsp;
[Issues](https://github.com/yakhyo/uniface/issues)

Runs on CPU, Apple Silicon and CUDA. Weights download on first use, verified by SHA-256.<br>
UniFace is [MIT](LICENSE); some pretrained weights are not, so check
[licences](https://yakhyo.github.io/uniface/license-attribution/) before shipping commercially.<br>
Not affiliated with [Uniface](https://uniface.com/) by Rocket Software.

</div>
