\[[English](README.md)\] | \[[Chinese](README.zh-CN.md)\] | \[[Korean](README.ko.md)\] | \[[Uzbek](README.uz.md)\]

<h1 align="center">UniFace: Python uchun yagona yuz tahlili kutubxonasi</h1>

Bu sahifa [inglizcha README](README.md) faylining oʻzbekcha tarjimasidir. Havolalardagi hujjatlar asosan ingliz tilida. Tarjimani yangilashda inglizcha nusxa bilan mosligini saqlang.

<div align="center">

[![PyPI Version](https://img.shields.io/pypi/v/uniface.svg?label=Version)](https://pypi.org/project/uniface/)
[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Github Build Status](https://github.com/yakhyo/uniface/actions/workflows/ci.yml/badge.svg)](https://github.com/yakhyo/uniface/actions)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/uniface?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=Downloads)](https://pepy.tech/projects/uniface)
[![Kaggle Badge](https://img.shields.io/badge/Notebooklar-Kaggle?label=Kaggle&color=blue)](https://www.kaggle.com/yakhyokhuja/code)
[![Hugging Face Spaces](https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Spaces-blue)](https://huggingface.co/spaces/yakhyo/uniface)

</div>

<div align="center">
    <img src="https://raw.githubusercontent.com/yakhyo/uniface/main/.github/logos/uniface_rounded_q80.webp" width="90%" alt="UniFace — Python uchun yagona yuz tahlili kutubxonasi">
</div>

<p align="center">
  UniFace — ishlab chiqarish muhitida foydalanishga tayyor, yengil Python kutubxonasi. U yuzlarni aniqlash, tanib olish,<br>
  kuzatish, tayanch nuqtalarni tahlil qilish, yuz qismlarini segmentlash, nigoh yoʻnalishini baholash va yuz atributlarini tahlil qilishni qoʻllab-quvvatlaydi.
</p>

<p align="center">
  <a href="https://yakhyo.github.io/uniface/quickstart/"><img src="https://img.shields.io/badge/Boshlash-1f6feb?style=for-the-badge&logoColor=white" alt="Boshlash"></a>
  &nbsp;
  <a href="https://yakhyo.github.io/uniface/models/"><img src="https://img.shields.io/badge/Modellar%20to%CA%BBplami-30363d?style=for-the-badge&logoColor=white" alt="Modellar toʻplami"></a>
  &nbsp;
  <a href="https://yakhyo.github.io/uniface/notebooks/"><img src="https://img.shields.io/badge/Notebooklar-30363d?style=for-the-badge&logo=jupyter&logoColor=white" alt="Notebooklar"></a>
  &nbsp;
  <a href="https://yakhyo.github.io/uniface/"><img src="https://img.shields.io/badge/To%CA%BBliq%20hujjatlar-30363d?style=for-the-badge&logoColor=white" alt="Toʻliq hujjatlar"></a>
</p>

```bash
pip install "uniface[cpu]"          # CPU va Apple Silicon
pip install "uniface[gpu]"          # NVIDIA CUDA
pip install --pre "uniface[cpu]"    # eng soʻnggi dastlabki reliz
```

### Birinchi skript

`FaceAnalyzer` bitta chaqiruvda yuzni aniqlash, tekislash va tanib olishni bajaradi. Atribut modellarini alohida qoʻshish kerak.

```python
import cv2
from uniface import FaceAnalyzer, FairFace

analyzer = FaceAnalyzer(predictors=[FairFace()])

for face in analyzer.analyze(cv2.imread("photo.jpg")):
    print(face.bbox, face.sex, face.age_group, face.embedding.shape)
```

`bbox`, `confidence`, `landmarks` va `embedding` har doim toʻldiriladi. Yosh, jins, irq, hissiyot, sifat
va yuz holatlari tegishli bashorat modelini uzatmaguningizcha `None` boʻlib qoladi.

<details>
<summary><b>Barcha oʻn beshta vazifa va ularni bajaradigan modellar</b></summary>

<br>

| Vazifa | Modellar |
| --- | --- |
| Yuzni aniqlash | RetinaFace, SCRFD, CenterFace, YOLOv5-Face, YOLOv8-Face, BlazeFace |
| Yuzni tanib olish | AdaFace, ArcFace, EdgeFace, MobileFace, SphereFace |
| Yuzni kuzatish | BYTETracker, video kadrlari orasida saqlanadigan ID lar |
| Yuz tayanch nuqtalari | 2d106det (106), PIPNet (98 / 68), Face Mesh (468 / 478, 3D) |
| Yuz qismlarini segmentlash | BiSeNet (19 ta sinf), XSeg yordamida niqob yaratish |
| Odam tasvirini fondan ajratish | MODNet, trimap talab qilmaydi |
| Nigoh yoʻnalishini baholash | MobileGaze (ResNet-18 / 34 / 50, MobileNetV2) |
| Bosh holati | 6D aylanish ifodasi, pitch / yaw / roll |
| Demografik atributlar | AgeGender, FairFace (yosh guruhi, jins, irq) |
| Hissiyot | AffectNet-7 va AffectNet-8 |
| Yuz holatlari | FaceAttribNet: koʻz holati, koʻzoynak, quyosh koʻzoynagi, niqob |
| Yuz tasviri sifati | eDifFIQA (T / S / M / L) |
| Soxta yuzlardan himoya | MiniFASNet yordamida tiriklikni tekshirish |
| Anonimlashtirish | 5 xil xiralashtirish usuli |
| Vektorlar ombori | FAISS asosida embedding vektorlarini qidirish |

CPU, Apple Silicon va CUDA da ishlaydi. Model vaznlari birinchi foydalanishda yuklab olinadi va SHA-256 orqali tekshiriladi.

</details>

<br>

### Yuzlarni topish va xususiyatlarini oʻlchash

**Yuzni aniqlash** &nbsp;·&nbsp; [hujjatlar](https://yakhyo.github.io/uniface/modules/detection/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/detection.jpg" width="100%">

**Yuz tayanch nuqtalari** &nbsp;·&nbsp; [hujjatlar](https://yakhyo.github.io/uniface/modules/landmarks/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/landmarks.jpg" width="100%">

**Yuz toʻri** &nbsp;·&nbsp; [hujjatlar](https://yakhyo.github.io/uniface/modules/landmarks/#face-mesh-468-or-478-points-3d)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/face_mesh.jpg" width="100%">

**Yuz tasviri sifati** &nbsp;·&nbsp; [hujjatlar](https://yakhyo.github.io/uniface/modules/quality/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/quality.jpg" width="100%">

### Yuz va odam tasvirini ajratib olish

**Yuz qismlarini segmentlash** &nbsp;·&nbsp; [hujjatlar](https://yakhyo.github.io/uniface/modules/parsing/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/parsing.jpg" width="100%">

**Yuzni segmentlash** &nbsp;·&nbsp; [hujjatlar](https://yakhyo.github.io/uniface/modules/parsing/#xseg)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/segmentation.jpg" width="100%">

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/segmentation_occluded.jpg" width="100%">

**Odam tasvirini fondan ajratish** &nbsp;·&nbsp; [hujjatlar](https://yakhyo.github.io/uniface/modules/matting/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/matting.jpg" width="100%">

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/matting_alt.jpg" width="100%">

### Bosh va nigoh yoʻnalishini baholash

**Bosh holati** &nbsp;·&nbsp; [hujjatlar](https://yakhyo.github.io/uniface/modules/headpose/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/headpose.jpg" width="100%">

**Nigoh yoʻnalishini baholash** &nbsp;·&nbsp; [hujjatlar](https://yakhyo.github.io/uniface/modules/gaze/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/gaze.jpg" width="100%">

### Yuz atributlarini tahlil qilish

**Yosh va jins** &nbsp;·&nbsp; [hujjatlar](https://yakhyo.github.io/uniface/modules/attributes/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/demography.jpg" width="100%">

**Hissiyot** &nbsp;·&nbsp; [hujjatlar](https://yakhyo.github.io/uniface/modules/attributes/#emotion)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/emotion.jpg" width="100%">

**Yuz holatlari** &nbsp;·&nbsp; [hujjatlar](https://yakhyo.github.io/uniface/modules/attributes/#faceattribnet)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/face_states.jpg" width="100%">

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/face_states_alt.jpg" width="100%">

### Haqiqiy yuzni qayta namoyish etilgan tasvirdan farqlash

**Soxta yuzlardan himoya** &nbsp;·&nbsp; [hujjatlar](https://yakhyo.github.io/uniface/modules/spoofing/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/spoofing.jpg" width="100%">

### Yuzlarni solishtirish yoki anonimlashtirish

**Yuzni tanib olish** &nbsp;·&nbsp; [hujjatlar](https://yakhyo.github.io/uniface/modules/recognition/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/verification.jpg" width="100%">

**Yuzni anonimlashtirish** &nbsp;·&nbsp; [hujjatlar](https://yakhyo.github.io/uniface/modules/privacy/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/anonymization.jpg" width="100%">

<br>

<div align="center">

**[Boshlash](https://yakhyo.github.io/uniface/quickstart/)** &nbsp;·&nbsp;
[Modellar toʻplami](https://yakhyo.github.io/uniface/models/) &nbsp;·&nbsp;
[Notebooklar](https://yakhyo.github.io/uniface/notebooks/) &nbsp;·&nbsp;
[Model litsenziyalari](https://yakhyo.github.io/uniface/license-attribution/) &nbsp;·&nbsp;
[Hissa qoʻshish](CONTRIBUTING.md) &nbsp;·&nbsp;
[Discord](https://discord.gg/wdzrjr7R5j) &nbsp;·&nbsp;
[Muammolar](https://github.com/yakhyo/uniface/issues)

CPU, Apple Silicon va CUDA da ishlaydi. Model vaznlari birinchi foydalanishda yuklab olinadi va SHA-256 orqali tekshiriladi.<br>
UniFace [MIT](LICENSE) litsenziyasi asosida tarqatiladi; ayrim oldindan oʻqitilgan model vaznlari boshqa litsenziyalarga ega. Tijoriy tarqatishdan oldin
[litsenziyalarni](https://yakhyo.github.io/uniface/license-attribution/) tekshiring.<br>
Loyiha Rocket Software kompaniyasining [Uniface](https://uniface.com/) mahsulotiga aloqador emas.

</div>
