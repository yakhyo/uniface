# Datasets

Overview of all training datasets and evaluation benchmarks used by UniFace models.

---

## Quick Reference

| Task        | Dataset                                          | Scale                  | Models                                      |
| ----------- | ------------------------------------------------ | ---------------------- | ------------------------------------------- |
| Detection   | [WIDER FACE](#wider-face)                        | 32K images             | RetinaFace, SCRFD, YOLOv5-Face, YOLOv8-Face |
| Recognition | [MS1MV2](#ms1mv2)                                | 5.8M images, 85.7K IDs | MobileFace, SphereFace                      |
| Recognition | [WebFace600K](#webface600k)                      | 600K images            | ArcFace                                     |
| Recognition | [WebFace4M / WebFace12M](#webface4m--webface12m) | 4M / 12M images        | AdaFace                                     |
| Gaze        | [Gaze360](#gaze360)                              | 238 subjects           | MobileGaze                                  |
| Parsing     | [CelebAMask-HQ](#celebamask-hq)                  | 30K images             | BiSeNet                                     |
| Attributes  | [CelebA](#celeba)                                | 200K images            | AgeGender                                   |
| Attributes  | [FairFace](#fairface)                            | Balanced demographics  | FairFace                                    |
| Attributes  | [AffectNet](#affectnet)                          | Emotion labels         | Emotion                                     |

---

## Training Datasets

### Face Detection

#### WIDER FACE

Large-scale face detection benchmark with images across 61 event categories. Contains faces with a high degree of variability in scale, pose, occlusion, expression, and illumination.

| Property | Value                                       |
| -------- | ------------------------------------------- |
| Images   | ~32,000 (train/val/test split)              |
| Faces    | ~394,000 annotated                          |
| Subsets  | Easy, Medium, Hard                          |
| Used by  | RetinaFace, SCRFD, YOLOv5-Face, YOLOv8-Face |

!!! info "Download & References"
**Paper**: [WIDER FACE: A Face Detection Benchmark](https://arxiv.org/abs/1511.06523)

    **Download**: [http://shuoyang1213.me/WIDERFACE/](http://shuoyang1213.me/WIDERFACE/)

---

### Face Recognition

#### MS1MV2

Refined version of the MS-Celeb-1M dataset, cleaned by InsightFace. Widely used for training face recognition models.

| Property   | Value                          |
| ---------- | ------------------------------ |
| Identities | 85.7K                          |
| Images     | 5.8M                           |
| Format     | Aligned and cropped to 112x112 |
| Used by    | MobileFace, SphereFace         |

!!! info "Download"
**Kaggle (aligned 112x112)**: [ms1m-arcface-dataset](https://www.kaggle.com/datasets/yakhyokhuja/ms1m-arcface-dataset) (from InsightFace)

    **Training code**: [yakhyo/face-recognition](https://github.com/yakhyo/face-recognition)

---

#### WebFace600K

Medium-scale face recognition dataset from the WebFace series.

| Property | Value   |
| -------- | ------- |
| Images   | ~600K   |
| Used by  | ArcFace |

!!! info "Source"
**Origin**: [InsightFace](https://github.com/deepinsight/insightface)

    **Paper**: [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)

---

#### WebFace4M / WebFace12M

Large-scale face recognition datasets from the WebFace260M collection. Used for training AdaFace models with adaptive quality-aware margin.

| Property | WebFace4M     | WebFace12M     |
| -------- | ------------- | -------------- |
| Images   | ~4M           | ~12M           |
| Used by  | AdaFace IR_18 | AdaFace IR_101 |

!!! info "Source"
**Paper**: [AdaFace: Quality Adaptive Margin for Face Recognition](https://arxiv.org/abs/2204.00964)

    **Original code**: [mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace)

---

#### CASIA-WebFace

Smaller-scale face recognition dataset suitable for academic research and lighter training runs.

| Property   | Value                          |
| ---------- | ------------------------------ |
| Identities | 10.6K                          |
| Images     | 491K                           |
| Format     | Aligned and cropped to 112x112 |
| Used by    | Alternative training set       |

!!! info "Download"
**Kaggle (aligned 112x112)**: [webface-112x112](https://www.kaggle.com/datasets/yakhyokhuja/webface-112x112) (from OpenSphere)

---

#### VGGFace2

Large-scale dataset with wide variations in pose, age, illumination, ethnicity, and profession.

| Property   | Value                          |
| ---------- | ------------------------------ |
| Identities | 8.6K                           |
| Images     | 3.1M                           |
| Format     | Aligned and cropped to 112x112 |
| Used by    | Alternative training set       |

!!! info "Download"
**Kaggle (aligned 112x112)**: [vggface2-112x112](https://www.kaggle.com/datasets/yakhyokhuja/vggface2-112x112) (from OpenSphere)

---

### Gaze Estimation

#### Gaze360

Large-scale gaze estimation dataset collected in indoor and outdoor environments with diverse head poses and wide gaze ranges (up to 360 degrees).

| Property    | Value                 |
| ----------- | --------------------- |
| Subjects    | 238                   |
| Environment | Indoor and outdoor    |
| Used by     | All MobileGaze models |

!!! info "Download & Preprocessing"
**Download**: [gaze360.csail.mit.edu/download.php](https://gaze360.csail.mit.edu/download.php)

    **Preprocessing**: [GazeHub - Gaze360](https://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/#gaze360)

!!! note "UniFace Models"
All MobileGaze models shipped with UniFace are trained exclusively on Gaze360 for 200 epochs.

**Dataset structure:**

```
data/
└── Gaze360/
    ├── Image/
    └── Label/
```

---

#### MPIIFaceGaze

Dataset for appearance-based gaze estimation from laptop webcam images of participants during everyday laptop usage. Supported by the gaze estimation training code but not used for the UniFace pretrained weights.

| Property    | Value                                    |
| ----------- | ---------------------------------------- |
| Subjects    | 15                                       |
| Environment | Everyday laptop usage                    |
| Used by     | Supported (not used for UniFace weights) |

!!! info "Download & Preprocessing"
**Download**: [MPIIFaceGaze download page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation)

    **Preprocessing**: [GazeHub - MPIIFaceGaze](https://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/#mpiifacegaze)

**Dataset structure:**

```
data/
└── MPIIFaceGaze/
    ├── Image/
    └── Label/
```

---

### Face Parsing

#### CelebAMask-HQ

High-quality face parsing dataset with pixel-level annotations for 19 facial component classes.

| Property   | Value                        |
| ---------- | ---------------------------- |
| Images     | 30,000                       |
| Classes    | 19 facial components         |
| Resolution | High quality                 |
| Used by    | BiSeNet (ResNet18, ResNet34) |

!!! info "Source"
**GitHub**: [switchablenorms/CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)

    **Training code**: [yakhyo/face-parsing](https://github.com/yakhyo/face-parsing)

**Dataset structure:**

```
dataset/
├── images/           # Input face images
│   ├── image1.jpg
│   └── ...
└── labels/           # Segmentation masks
    ├── image1.png
    └── ...
```

---

### Attribute Analysis

#### CelebA

Large-scale face attributes dataset widely used for training age and gender prediction models.

| Property   | Value                |
| ---------- | -------------------- |
| Images     | ~200K                |
| Attributes | 40 binary attributes |
| Used by    | AgeGender            |

!!! info "Reference"
**Paper**: [Deep Learning Face Attributes in the Wild](https://arxiv.org/abs/1411.7766)

---

#### FairFace

Face attribute dataset designed for balanced representation across race, gender, and age groups. Provides more equitable predictions compared to imbalanced datasets.

| Property   | Value                               |
| ---------- | ----------------------------------- |
| Attributes | Race (7), Gender (2), Age Group (9) |
| Used by    | FairFace                            |
| License    | CC BY 4.0                           |

!!! info "Reference"
**Paper**: [FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age](https://arxiv.org/abs/1908.04913)

    **ONNX inference**: [yakhyo/fairface-onnx](https://github.com/yakhyo/fairface-onnx)

---

#### AffectNet

Large-scale facial expression dataset for emotion recognition training.

| Property | Value                                                                   |
| -------- | ----------------------------------------------------------------------- |
| Classes  | 7 or 8 (Neutral, Happy, Sad, Surprise, Fear, Disgust, Angry + Contempt) |
| Used by  | Emotion (AFFECNET7, AFFECNET8)                                          |

!!! info "Reference"
**Paper**: [AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild](https://ieeexplore.ieee.org/document/8013713)

---

## Evaluation Benchmarks

### Face Detection

#### WIDER FACE Validation Set

The standard benchmark for face detection models. Results are reported across three difficulty subsets.

| Subset | Criteria                                      |
| ------ | --------------------------------------------- |
| Easy   | Large, clear, unoccluded faces                |
| Medium | Moderate scale and occlusion                  |
| Hard   | Small, heavily occluded, or challenging faces |

See [Model Zoo - Detection](models.md#face-detection-models) for per-model accuracy on each subset.

---

### Face Recognition

Recognition models are evaluated across multiple benchmarks. Aligned 112x112 validation datasets are available as a single download.

!!! info "Download"
**Kaggle**: [agedb-30-calfw-cplfw-lfw-aligned-112x112](https://www.kaggle.com/datasets/yakhyokhuja/agedb-30-calfw-cplfw-lfw-aligned-112x112)

| Benchmark    | Description                                                       | Used by                         |
| ------------ | ----------------------------------------------------------------- | ------------------------------- |
| **LFW**      | Labeled Faces in the Wild - standard face verification benchmark  | ArcFace, MobileFace, SphereFace |
| **CALFW**    | Cross-Age LFW - face verification across age gaps                 | MobileFace, SphereFace          |
| **CPLFW**    | Cross-Pose LFW - face verification across pose variations         | MobileFace, SphereFace          |
| **AgeDB-30** | Age database with 30-year age gaps                                | ArcFace, MobileFace, SphereFace |
| **CFP-FP**   | Celebrities in Frontal-Profile - frontal vs. profile verification | ArcFace                         |
| **IJB-B**    | IARPA Janus Benchmark B - TAR@FAR=0.01%                           | AdaFace                         |
| **IJB-C**    | IARPA Janus Benchmark C - TAR@FAR=1e-4                            | AdaFace, ArcFace                |

See [Model Zoo - Recognition](models.md#face-recognition-models) for per-model accuracy on each benchmark.

---

### Gaze Estimation

| Benchmark            | Metric        | Description                                  |
| -------------------- | ------------- | -------------------------------------------- |
| **Gaze360 test set** | MAE (degrees) | Mean Absolute Error in gaze angle prediction |

See [Model Zoo - Gaze](models.md#gaze-estimation-models) for per-model MAE scores.

---

## Training Repositories

For training your own models or reproducing results, see the following repositories:

| Task        | Repository                                                                | Datasets Supported              |
| ----------- | ------------------------------------------------------------------------- | ------------------------------- |
| Detection   | [yakhyo/retinaface-pytorch](https://github.com/yakhyo/retinaface-pytorch) | WIDER FACE                      |
| Recognition | [yakhyo/face-recognition](https://github.com/yakhyo/face-recognition)     | MS1MV2, CASIA-WebFace, VGGFace2 |
| Gaze        | [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation)       | Gaze360, MPIIFaceGaze           |
| Parsing     | [yakhyo/face-parsing](https://github.com/yakhyo/face-parsing)             | CelebAMask-HQ                   |
