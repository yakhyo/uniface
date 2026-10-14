\[[English](README.md)\] | \[[Chinese](README.zh-CN.md)\] | \[[Korean](README.ko.md)\] | \[[Spanish](README.es.md)\] | \[[Uzbek](README.uz.md)\]

<h1 align="center">UniFace: Una biblioteca unificada de análisis facial para Python</h1>

Esta página es la traducción al español del [README en inglés](README.md). La documentación enlazada está principalmente en inglés. Al actualizar la traducción, mantén el contenido sincronizado con la versión inglesa.

<div align="center">

[![PyPI Version](https://img.shields.io/pypi/v/uniface.svg?label=Version)](https://pypi.org/project/uniface/)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Github Build Status](https://github.com/yakhyo/uniface/actions/workflows/ci.yml/badge.svg)](https://github.com/yakhyo/uniface/actions)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/uniface?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=Downloads)](https://pepy.tech/projects/uniface)
[![Kaggle Badge](https://img.shields.io/badge/Cuadernos-Kaggle?label=Kaggle&color=blue)](https://www.kaggle.com/yakhyokhuja/code)
[![Hugging Face Spaces](https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Spaces-blue)](https://huggingface.co/spaces/yakhyo/uniface)

</div>

<div align="center">
    <img src="https://raw.githubusercontent.com/yakhyo/uniface/main/.github/logos/uniface_rounded_q80.webp" width="90%" alt="UniFace — Una biblioteca unificada de análisis facial para Python">
</div>

<p align="center">
  UniFace es una biblioteca ligera de Python, lista para producción, para detección y reconocimiento facial,<br>
  seguimiento, análisis de puntos de referencia, segmentación de partes del rostro, estimación de la mirada y atributos faciales.
</p>

<p align="center">
  <a href="https://yakhyo.github.io/uniface/quickstart/"><img src="https://img.shields.io/badge/Primeros%20pasos-1f6feb?style=for-the-badge&logoColor=white" alt="Primeros pasos"></a>
  &nbsp;
  <a href="https://yakhyo.github.io/uniface/models/"><img src="https://img.shields.io/badge/Cat%C3%A1logo%20de%20modelos-30363d?style=for-the-badge&logoColor=white" alt="Catálogo de modelos"></a>
  &nbsp;
  <a href="https://yakhyo.github.io/uniface/notebooks/"><img src="https://img.shields.io/badge/Cuadernos-30363d?style=for-the-badge&logo=jupyter&logoColor=white" alt="Cuadernos"></a>
  &nbsp;
  <a href="https://yakhyo.github.io/uniface/"><img src="https://img.shields.io/badge/Documentaci%C3%B3n%20completa-30363d?style=for-the-badge&logoColor=white" alt="Documentación completa"></a>
</p>

```bash
pip install "uniface[cpu]"          # CPU y Apple Silicon
pip install "uniface[gpu]"          # NVIDIA CUDA
pip install --pre "uniface[cpu]"    # última versión preliminar
```

<details>
<summary><b>Un primer script</b></summary>

<br>

`FaceAnalyzer` ejecuta la detección, la alineación y el reconocimiento en una sola llamada. Los modelos de atributos se añaden de forma explícita.

```python
import cv2
from uniface import FaceAnalyzer, FairFace

analyzer = FaceAnalyzer(predictors=[FairFace()])

for face in analyzer.analyze(cv2.imread("photo.jpg")):
    print(face.bbox, face.sex, face.age_group, face.embedding.shape)
```

`bbox`, `confidence`, `landmarks` y `embedding` siempre tienen un valor. La edad, el sexo, la raza, la emoción, la calidad
y los estados del rostro permanecen en `None` hasta que se añade el predictor correspondiente.

</details>

<details>
<summary><b>Las quince tareas y sus modelos</b></summary>

<br>

| Tarea | Modelos |
| --- | --- |
| Detección facial | RetinaFace, SCRFD, CenterFace, YOLOv5-Face, YOLOv8-Face, BlazeFace |
| Reconocimiento facial | AdaFace, ArcFace, EdgeFace, MobileFace, SphereFace |
| Seguimiento facial | BYTETracker, identificadores persistentes entre fotogramas |
| Puntos de referencia faciales | 2d106det (106), PIPNet (98 / 68), Face Mesh (468 / 478, 3D) |
| Segmentación de partes del rostro | BiSeNet (19 clases), máscaras con XSeg |
| Recorte de retratos | MODNet, sin trimap |
| Estimación de la mirada | MobileGaze (ResNet-18 / 34 / 50, MobileNetV2) |
| Pose de la cabeza | representación de rotación 6D, cabeceo / guiñada / alabeo |
| Atributos demográficos | AgeGender, FairFace (grupo de edad, sexo, raza) |
| Emoción | AffectNet-7 y AffectNet-8 |
| Estados del rostro | FaceAttribNet: estado de los ojos, gafas, gafas de sol, mascarilla |
| Calidad facial | eDifFIQA (T / S / M / L) |
| Detección de suplantación facial | detección de vida con MiniFASNet |
| Anonimización | 5 métodos de desenfoque |
| Almacén de vectores | búsqueda de embeddings con FAISS |

Funciona en CPU, Apple Silicon y CUDA. Los pesos se descargan al usarlos por primera vez y se verifican con SHA-256.

</details>

<br>

### Detectar rostros y medir sus características

**Detección facial** &nbsp;·&nbsp; [documentación](https://yakhyo.github.io/uniface/modules/detection/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/detection.jpg" width="100%">

**Puntos de referencia faciales** &nbsp;·&nbsp; [documentación](https://yakhyo.github.io/uniface/modules/landmarks/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/landmarks.jpg" width="100%">

**Malla facial** &nbsp;·&nbsp; [documentación](https://yakhyo.github.io/uniface/modules/landmarks/#face-mesh-468-or-478-points-3d)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/face_mesh.jpg" width="100%">

**Calidad facial** &nbsp;·&nbsp; [documentación](https://yakhyo.github.io/uniface/modules/quality/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/quality.jpg" width="100%">

### Separar rostros y retratos

**Segmentación de partes del rostro** &nbsp;·&nbsp; [documentación](https://yakhyo.github.io/uniface/modules/parsing/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/parsing.jpg" width="100%">

**Segmentación facial** &nbsp;·&nbsp; [documentación](https://yakhyo.github.io/uniface/modules/parsing/#xseg)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/segmentation.jpg" width="100%">

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/segmentation_occluded.jpg" width="100%">

**Recorte de retratos** &nbsp;·&nbsp; [documentación](https://yakhyo.github.io/uniface/modules/matting/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/matting.jpg" width="100%">

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/matting_alt.jpg" width="100%">

### Estimar la orientación de la cabeza y la mirada

**Pose de la cabeza** &nbsp;·&nbsp; [documentación](https://yakhyo.github.io/uniface/modules/headpose/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/headpose.jpg" width="100%">

**Estimación de la mirada** &nbsp;·&nbsp; [documentación](https://yakhyo.github.io/uniface/modules/gaze/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/gaze.jpg" width="100%">

### Analizar atributos faciales

**Edad y sexo** &nbsp;·&nbsp; [documentación](https://yakhyo.github.io/uniface/modules/attributes/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/demography.jpg" width="100%">

**Emoción** &nbsp;·&nbsp; [documentación](https://yakhyo.github.io/uniface/modules/attributes/#emotion)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/emotion.jpg" width="100%">

**Estados del rostro** &nbsp;·&nbsp; [documentación](https://yakhyo.github.io/uniface/modules/attributes/#faceattribnet)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/face_states.jpg" width="100%">

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/face_states_alt.jpg" width="100%">

### Distinguir un rostro real de una reproducción

**Detección de suplantación facial** &nbsp;·&nbsp; [documentación](https://yakhyo.github.io/uniface/modules/spoofing/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/spoofing.jpg" width="100%">

### Comparar rostros o anonimizarlos

**Reconocimiento facial** &nbsp;·&nbsp; [documentación](https://yakhyo.github.io/uniface/modules/recognition/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/verification.jpg" width="100%">

**Anonimización facial** &nbsp;·&nbsp; [documentación](https://yakhyo.github.io/uniface/modules/privacy/)

<img src="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/demo/anonymization.jpg" width="100%">

<br>

<div align="center">

**[Primeros pasos](https://yakhyo.github.io/uniface/quickstart/)** &nbsp;·&nbsp;
[Catálogo de modelos](https://yakhyo.github.io/uniface/models/) &nbsp;·&nbsp;
[Cuadernos](https://yakhyo.github.io/uniface/notebooks/) &nbsp;·&nbsp;
[Licencias de los modelos](https://yakhyo.github.io/uniface/license-attribution/) &nbsp;·&nbsp;
[Contribuir](CONTRIBUTING.md) &nbsp;·&nbsp;
[Discord](https://discord.gg/wdzrjr7R5j) &nbsp;·&nbsp;
[Incidencias](https://github.com/yakhyo/uniface/issues)

Funciona en CPU, Apple Silicon y CUDA. Los pesos se descargan al usarlos por primera vez y se verifican con SHA-256.<br>
UniFace se distribuye bajo la licencia [MIT](LICENSE); algunos pesos preentrenados tienen otras licencias, así que consulta las
[licencias](https://yakhyo.github.io/uniface/license-attribution/) antes de distribuirlo comercialmente.<br>
Este proyecto no está afiliado a [Uniface](https://uniface.com/) de Rocket Software.

</div>
