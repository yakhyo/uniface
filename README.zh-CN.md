\[[English](README.md)\] | \[[Chinese](README.zh-CN.md)\] | \[[Korean](README.ko.md)\] | \[[Uzbek](README.uz.md)\]

<h1 align="center">UniFace：统一的 Python 人脸分析库</h1>

本页是 [英文 README](README.md) 的简体中文翻译。链接指向的文档主要使用英文；更新翻译时请与英文版本保持同步。

<div align="center">

[![PyPI Version](https://img.shields.io/pypi/v/uniface.svg?label=Version)](https://pypi.org/project/uniface/)
[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Github Build Status](https://github.com/yakhyo/uniface/actions/workflows/ci.yml/badge.svg)](https://github.com/yakhyo/uniface/actions)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/uniface?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=Downloads)](https://pepy.tech/projects/uniface)
[![Kaggle Badge](https://img.shields.io/badge/%E4%BA%A4%E4%BA%92%E5%BC%8F%E7%AC%94%E8%AE%B0%E6%9C%AC-Kaggle?label=Kaggle&color=blue)](https://www.kaggle.com/yakhyokhuja/code)
[![Hugging Face Spaces](https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Spaces-blue)](https://huggingface.co/spaces/yakhyo/uniface)

</div>

<div align="center">
    <img src="https://raw.githubusercontent.com/yakhyo/uniface/main/.github/logos/uniface_rounded_q80.webp" width="90%" alt="UniFace — 统一的 Python 人脸分析库">
</div>

<p align="center">
  UniFace 是一个轻量级、可用于生产环境的 Python 库，支持人脸检测、识别、<br>
  跟踪、关键点分析、人脸解析、视线估计和人脸属性分析。
</p>

<p align="center">
  <a href="https://yakhyo.github.io/uniface/quickstart/"><img src="https://img.shields.io/badge/%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B-1f6feb?style=for-the-badge&logoColor=white" alt="快速开始"></a>
  &nbsp;
  <a href="https://yakhyo.github.io/uniface/models/"><img src="https://img.shields.io/badge/%E6%A8%A1%E5%9E%8B%E5%BA%93-30363d?style=for-the-badge&logoColor=white" alt="模型库"></a>
  &nbsp;
  <a href="https://yakhyo.github.io/uniface/notebooks/"><img src="https://img.shields.io/badge/%E4%BA%A4%E4%BA%92%E5%BC%8F%E7%AC%94%E8%AE%B0%E6%9C%AC-30363d?style=for-the-badge&logo=jupyter&logoColor=white" alt="交互式笔记本"></a>
  &nbsp;
  <a href="https://yakhyo.github.io/uniface/"><img src="https://img.shields.io/badge/%E5%AE%8C%E6%95%B4%E6%96%87%E6%A1%A3-30363d?style=for-the-badge&logoColor=white" alt="完整文档"></a>
</p>

```bash
pip install "uniface[cpu]"          # CPU 和 Apple Silicon
pip install "uniface[gpu]"          # NVIDIA CUDA
pip install --pre "uniface[cpu]"    # 最新预发布版本
```

### 第一个示例脚本

`FaceAnalyzer` 通过一次调用完成检测、对齐和识别。属性模型需要显式启用。

```python
import cv2
from uniface import FaceAnalyzer, FairFace

analyzer = FaceAnalyzer(predictors=[FairFace()])

for face in analyzer.analyze(cv2.imread("photo.jpg")):
    print(face.bbox, face.sex, face.age_group, face.embedding.shape)
```

`bbox`、`confidence`、`landmarks` 和 `embedding` 始终会被赋值。年龄、性别、种族、情绪、质量
和人脸状态保持为 `None`，直到你传入相应的预测模型。

<details>
<summary><b>全部十五项任务及其对应模型</b></summary>

<br>

| 任务 | 模型 |
| --- | --- |
| 人脸检测 | RetinaFace, SCRFD, CenterFace, YOLOv5-Face, YOLOv8-Face, BlazeFace |
| 人脸识别 | AdaFace, ArcFace, EdgeFace, MobileFace, SphereFace |
| 人脸跟踪 | BYTETracker, 视频帧之间保持一致的 ID |
| 人脸关键点 | 2d106det (106), PIPNet (98 / 68), Face Mesh (468 / 478, 3D) |
| 人脸解析 | BiSeNet (19 个类别), XSeg 掩码生成 |
| 人像抠图 | MODNet, 无需三分图 |
| 视线估计 | MobileGaze (ResNet-18 / 34 / 50, MobileNetV2) |
| 头部姿态 | 6D 旋转表示，俯仰 / 偏航 / 翻滚 |
| 人口统计属性 | AgeGender, FairFace (年龄组、性别、种族) |
| 情绪 | AffectNet-7 和 AffectNet-8 |
| 人脸状态 | FaceAttribNet: 眼部状态、眼镜、太阳镜、口罩 |
| 人脸质量 | eDifFIQA (T / S / M / L) |
| 人脸防伪 | MiniFASNet 活体检测 |
| 匿名化 | 5 种模糊方法 |
| 向量存储 | 基于 FAISS 的嵌入向量搜索 |

支持 CPU、Apple Silicon 和 CUDA。模型权重在首次使用时下载，并通过 SHA-256 校验。

</details>

<br>

### 检测人脸并测量特征

**人脸检测** &nbsp;·&nbsp; [文档](https://yakhyo.github.io/uniface/modules/detection/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/detection.jpg" width="100%">

**人脸关键点** &nbsp;·&nbsp; [文档](https://yakhyo.github.io/uniface/modules/landmarks/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/landmarks.jpg" width="100%">

**人脸网格** &nbsp;·&nbsp; [文档](https://yakhyo.github.io/uniface/modules/landmarks/#face-mesh-468-or-478-points-3d)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/face_mesh.jpg" width="100%">

**人脸质量** &nbsp;·&nbsp; [文档](https://yakhyo.github.io/uniface/modules/quality/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/quality.jpg" width="100%">

### 分割人脸与人像

**人脸解析** &nbsp;·&nbsp; [文档](https://yakhyo.github.io/uniface/modules/parsing/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/parsing.jpg" width="100%">

**人脸分割** &nbsp;·&nbsp; [文档](https://yakhyo.github.io/uniface/modules/parsing/#xseg)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/segmentation.jpg" width="100%">

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/segmentation_occluded.jpg" width="100%">

**人像抠图** &nbsp;·&nbsp; [文档](https://yakhyo.github.io/uniface/modules/matting/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/matting.jpg" width="100%">

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/matting_alt.jpg" width="100%">

### 估计头部朝向与视线方向

**头部姿态** &nbsp;·&nbsp; [文档](https://yakhyo.github.io/uniface/modules/headpose/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/headpose.jpg" width="100%">

**视线估计** &nbsp;·&nbsp; [文档](https://yakhyo.github.io/uniface/modules/gaze/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/gaze.jpg" width="100%">

### 分析人脸属性

**年龄与性别** &nbsp;·&nbsp; [文档](https://yakhyo.github.io/uniface/modules/attributes/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/demography.jpg" width="100%">

**情绪** &nbsp;·&nbsp; [文档](https://yakhyo.github.io/uniface/modules/attributes/#emotion)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/emotion.jpg" width="100%">

**人脸状态** &nbsp;·&nbsp; [文档](https://yakhyo.github.io/uniface/modules/attributes/#faceattribnet)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/face_states.jpg" width="100%">

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/face_states_alt.jpg" width="100%">

### 区分真实人脸与回放攻击

**人脸防伪** &nbsp;·&nbsp; [文档](https://yakhyo.github.io/uniface/modules/spoofing/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/spoofing.jpg" width="100%">

### 匹配人脸或进行匿名化处理

**人脸识别** &nbsp;·&nbsp; [文档](https://yakhyo.github.io/uniface/modules/recognition/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/verification.jpg" width="100%">

**人脸匿名化** &nbsp;·&nbsp; [文档](https://yakhyo.github.io/uniface/modules/privacy/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/anonymization.jpg" width="100%">

<br>

<div align="center">

**[快速开始](https://yakhyo.github.io/uniface/quickstart/)** &nbsp;·&nbsp;
[模型库](https://yakhyo.github.io/uniface/models/) &nbsp;·&nbsp;
[交互式笔记本](https://yakhyo.github.io/uniface/notebooks/) &nbsp;·&nbsp;
[模型许可证](https://yakhyo.github.io/uniface/license-attribution/) &nbsp;·&nbsp;
[参与贡献](CONTRIBUTING.md) &nbsp;·&nbsp;
[Discord](https://discord.gg/wdzrjr7R5j) &nbsp;·&nbsp;
[问题反馈](https://github.com/yakhyo/uniface/issues)

支持 CPU、Apple Silicon 和 CUDA。模型权重在首次使用时下载，并通过 SHA-256 校验。<br>
UniFace 使用 [MIT](LICENSE) 许可证；部分预训练权重采用其他许可证，因此在商业发布前请查看
[许可证说明](https://yakhyo.github.io/uniface/license-attribution/)。<br>
本项目与 Rocket Software 的 [Uniface](https://uniface.com/) 无关。

</div>
