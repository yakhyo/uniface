\[[English](README.md)\] | \[[Chinese](README.zh-CN.md)\] | \[[Korean](README.ko.md)\] | \[[Spanish](README.es.md)\] | \[[Uzbek](README.uz.md)\]

<h1 align="center">UniFace: Python용 통합 얼굴 분석 라이브러리</h1>

이 페이지는 [영문 README](README.md)의 한국어 번역입니다. 링크된 문서는 주로 영어로 제공됩니다. 번역을 수정할 때는 영문 버전과 내용을 맞춰 주세요.

<div align="center">

[![PyPI Version](https://img.shields.io/pypi/v/uniface.svg?label=Version)](https://pypi.org/project/uniface/)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Github Build Status](https://github.com/yakhyo/uniface/actions/workflows/ci.yml/badge.svg)](https://github.com/yakhyo/uniface/actions)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/uniface?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=Downloads)](https://pepy.tech/projects/uniface)
[![Kaggle Badge](https://img.shields.io/badge/%EB%85%B8%ED%8A%B8%EB%B6%81-Kaggle?label=Kaggle&color=blue)](https://www.kaggle.com/yakhyokhuja/code)
[![Hugging Face Spaces](https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Spaces-blue)](https://huggingface.co/spaces/yakhyo/uniface)

</div>

<div align="center">
    <img src="https://raw.githubusercontent.com/yakhyo/uniface/main/.github/logos/uniface_rounded_q80.webp" width="90%" alt="UniFace — Python용 통합 얼굴 분석 라이브러리">
</div>

<p align="center">
  UniFace는 가볍고 프로덕션 환경에서 사용할 수 있는 Python 라이브러리로, 얼굴 검출, 인식,<br>
  추적, 랜드마크 분석, 얼굴 파싱, 시선 추정 및 얼굴 속성 분석을 지원합니다.
</p>

<p align="center">
  <a href="https://yakhyo.github.io/uniface/quickstart/"><img src="https://img.shields.io/badge/%EC%8B%9C%EC%9E%91%ED%95%98%EA%B8%B0-1f6feb?style=for-the-badge&logoColor=white" alt="시작하기"></a>
  &nbsp;
  <a href="https://yakhyo.github.io/uniface/models/"><img src="https://img.shields.io/badge/%EB%AA%A8%EB%8D%B8%20%EB%AA%A8%EC%9D%8C-30363d?style=for-the-badge&logoColor=white" alt="모델 모음"></a>
  &nbsp;
  <a href="https://yakhyo.github.io/uniface/notebooks/"><img src="https://img.shields.io/badge/%EB%85%B8%ED%8A%B8%EB%B6%81-30363d?style=for-the-badge&logo=jupyter&logoColor=white" alt="노트북"></a>
  &nbsp;
  <a href="https://yakhyo.github.io/uniface/"><img src="https://img.shields.io/badge/%EC%A0%84%EC%B2%B4%20%EB%AC%B8%EC%84%9C-30363d?style=for-the-badge&logoColor=white" alt="전체 문서"></a>
</p>

```bash
pip install "uniface[cpu]"          # CPU 및 Apple Silicon
pip install "uniface[gpu]"          # NVIDIA CUDA
pip install --pre "uniface[cpu]"    # 최신 사전 릴리스
```

<details>
<summary><b>첫 번째 예제</b></summary>

<br>

`FaceAnalyzer`는 한 번의 호출로 검출, 정렬 및 인식을 수행합니다. 속성 모델은 필요할 때 명시적으로 추가합니다.

```python
import cv2
from uniface import FaceAnalyzer, FairFace

analyzer = FaceAnalyzer(predictors=[FairFace()])

for face in analyzer.analyze(cv2.imread("photo.jpg")):
    print(face.bbox, face.sex, face.age_group, face.embedding.shape)
```

`bbox`, `confidence`, `landmarks`, `embedding`은 항상 설정됩니다. 나이, 성별, 인종, 감정, 품질
및 얼굴 상태는 해당 값을 채우는 예측 모델을 전달하기 전까지 `None`으로 유지됩니다.

</details>

<details>
<summary><b>15가지 작업과 각 작업에 사용되는 모델</b></summary>

<br>

| 작업 | 모델 |
| --- | --- |
| 얼굴 검출 | RetinaFace, SCRFD, CenterFace, YOLOv5-Face, YOLOv8-Face, BlazeFace |
| 얼굴 인식 | AdaFace, ArcFace, EdgeFace, MobileFace, SphereFace |
| 얼굴 추적 | BYTETracker, 동영상 프레임 간 ID 유지 |
| 얼굴 랜드마크 | 2d106det (106), PIPNet (98 / 68), Face Mesh (468 / 478, 3D) |
| 얼굴 파싱 | BiSeNet (19개 클래스), XSeg 마스크 생성 |
| 인물 매팅 | MODNet, 트라이맵 불필요 |
| 시선 추정 | MobileGaze (ResNet-18 / 34 / 50, MobileNetV2) |
| 머리 자세 | 6D 회전 표현, 피치 / 요 / 롤 |
| 인구통계학적 속성 | AgeGender, FairFace (연령대, 성별, 인종) |
| 감정 | AffectNet-7 및 AffectNet-8 |
| 얼굴 상태 | FaceAttribNet: 눈 상태, 안경, 선글라스, 마스크 |
| 얼굴 품질 | eDifFIQA (T / S / M / L) |
| 얼굴 위조 방지 | MiniFASNet 생체 감지 |
| 익명화 | 5가지 블러 방식 |
| 벡터 저장소 | FAISS 기반 임베딩 검색 |

CPU, Apple Silicon 및 CUDA에서 실행됩니다. 모델 가중치는 처음 사용할 때 다운로드되며 SHA-256으로 검증됩니다.

</details>

<br>

### 얼굴 검출과 특징 측정

**얼굴 검출** &nbsp;·&nbsp; [문서](https://yakhyo.github.io/uniface/modules/detection/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/detection.jpg" width="100%">

**얼굴 랜드마크** &nbsp;·&nbsp; [문서](https://yakhyo.github.io/uniface/modules/landmarks/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/landmarks.jpg" width="100%">

**얼굴 메시** &nbsp;·&nbsp; [문서](https://yakhyo.github.io/uniface/modules/landmarks/#face-mesh-468-or-478-points-3d)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/face_mesh.jpg" width="100%">

**얼굴 품질** &nbsp;·&nbsp; [문서](https://yakhyo.github.io/uniface/modules/quality/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/quality.jpg" width="100%">

### 얼굴과 인물 영역 분리

**얼굴 파싱** &nbsp;·&nbsp; [문서](https://yakhyo.github.io/uniface/modules/parsing/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/parsing.jpg" width="100%">

**얼굴 세그멘테이션** &nbsp;·&nbsp; [문서](https://yakhyo.github.io/uniface/modules/parsing/#xseg)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/segmentation.jpg" width="100%">

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/segmentation_occluded.jpg" width="100%">

**인물 매팅** &nbsp;·&nbsp; [문서](https://yakhyo.github.io/uniface/modules/matting/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/matting.jpg" width="100%">

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/matting_alt.jpg" width="100%">

### 머리 방향과 시선 추정

**머리 자세** &nbsp;·&nbsp; [문서](https://yakhyo.github.io/uniface/modules/headpose/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/headpose.jpg" width="100%">

**시선 추정** &nbsp;·&nbsp; [문서](https://yakhyo.github.io/uniface/modules/gaze/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/gaze.jpg" width="100%">

### 얼굴 속성 분석

**나이와 성별** &nbsp;·&nbsp; [문서](https://yakhyo.github.io/uniface/modules/attributes/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/demography.jpg" width="100%">

**감정** &nbsp;·&nbsp; [문서](https://yakhyo.github.io/uniface/modules/attributes/#emotion)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/emotion.jpg" width="100%">

**얼굴 상태** &nbsp;·&nbsp; [문서](https://yakhyo.github.io/uniface/modules/attributes/#faceattribnet)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/face_states.jpg" width="100%">

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/face_states_alt.jpg" width="100%">

### 실제 얼굴과 재생 공격 구분

**얼굴 위조 방지** &nbsp;·&nbsp; [문서](https://yakhyo.github.io/uniface/modules/spoofing/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/spoofing.jpg" width="100%">

### 얼굴 매칭과 익명화

**얼굴 인식** &nbsp;·&nbsp; [문서](https://yakhyo.github.io/uniface/modules/recognition/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/verification.jpg" width="100%">

**얼굴 익명화** &nbsp;·&nbsp; [문서](https://yakhyo.github.io/uniface/modules/privacy/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/anonymization.jpg" width="100%">

<br>

<div align="center">

**[시작하기](https://yakhyo.github.io/uniface/quickstart/)** &nbsp;·&nbsp;
[모델 모음](https://yakhyo.github.io/uniface/models/) &nbsp;·&nbsp;
[노트북](https://yakhyo.github.io/uniface/notebooks/) &nbsp;·&nbsp;
[모델 라이선스](https://yakhyo.github.io/uniface/license-attribution/) &nbsp;·&nbsp;
[기여하기](CONTRIBUTING.md) &nbsp;·&nbsp;
[Discord](https://discord.gg/wdzrjr7R5j) &nbsp;·&nbsp;
[이슈](https://github.com/yakhyo/uniface/issues)

CPU, Apple Silicon 및 CUDA에서 실행됩니다. 모델 가중치는 처음 사용할 때 다운로드되며 SHA-256으로 검증됩니다.<br>
UniFace는 [MIT](LICENSE) 라이선스를 사용합니다. 일부 사전 학습 가중치에는 다른 라이선스가 적용되므로 상업적으로 배포하기 전에
[라이선스 안내](https://yakhyo.github.io/uniface/license-attribution/)를 확인하세요.<br>
이 프로젝트는 Rocket Software의 [Uniface](https://uniface.com/)와 관련이 없습니다.

</div>
