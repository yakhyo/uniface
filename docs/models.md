# Model Zoo

Complete guide to all available models and their performance characteristics.

---

## Face Detection Models

### RetinaFace Family

RetinaFace models are trained on the [WIDER FACE](datasets.md#wider-face) dataset.

| Model Name     | Params | Size  | Easy   | Medium | Hard   |
| -------------- | ------ | ----- | ------ | ------ | ------ |
| `MNET_025`   | 0.4M   | 1.7MB | 88.48% | 87.02% | 80.61% |
| `MNET_050`   | 1.0M   | 2.6MB | 89.42% | 87.97% | 82.40% |
| `MNET_V1`    | 3.5M   | 3.8MB | 90.59% | 89.14% | 84.13% |
| `MNET_V2` :material-check-circle: | 3.2M   | 3.5MB | 91.70% | 91.03% | 86.60% |
| `RESNET18`   | 11.7M  | 27MB  | 92.50% | 91.02% | 86.63% |
| `RESNET34`   | 24.8M  | 56MB  | 94.16% | 93.12% | 88.90% |

!!! info "Accuracy & Benchmarks"
    **Accuracy**: WIDER FACE validation set (Easy/Medium/Hard subsets) - from [RetinaFace paper](https://arxiv.org/abs/1905.00641)

    **Speed**: Benchmark on your own hardware using `python tools/detect.py --source <image>`

---

### SCRFD Family

SCRFD (Sample and Computation Redistribution for Efficient Face Detection) models trained on [WIDER FACE](datasets.md#wider-face) dataset.

| Model Name       | Params | Size  | Easy   | Medium | Hard   |
| ---------------- | ------ | ----- | ------ | ------ | ------ |
| `SCRFD_500M_KPS`   | 0.6M   | 2.5MB | 90.57% | 88.12% | 68.51% |
| `SCRFD_10G_KPS` :material-check-circle: | 4.2M   | 17MB  | 95.16% | 93.87% | 83.05% |

!!! info "Accuracy & Benchmarks"
    **Accuracy**: WIDER FACE validation set - from [SCRFD paper](https://arxiv.org/abs/2105.04714)

    **Speed**: Benchmark on your own hardware using `python tools/detect.py --source <image>`

---

### YOLOv5-Face Family

YOLOv5-Face models provide detection with 5-point facial landmarks, trained on [WIDER FACE](datasets.md#wider-face) dataset.

| Model Name     | Size | Easy   | Medium | Hard   |
| -------------- | ---- | ------ | ------ | ------ |
| `YOLOV5N`    | 11MB | 93.61% | 91.52% | 80.53% |
| `YOLOV5S` :material-check-circle: | 28MB | 94.33% | 92.61% | 83.15% |
| `YOLOV5M`    | 82MB | 95.30% | 93.76% | 85.28% |

!!! info "Accuracy & Benchmarks"
    **Accuracy**: WIDER FACE validation set - from [YOLOv5-Face paper](https://arxiv.org/abs/2105.12931)

    **Speed**: Benchmark on your own hardware using `python tools/detect.py --source <image>`

!!! note "Fixed Input Size"
    All YOLOv5-Face models use a fixed input size of 640×640.

---

### YOLOv8-Face Family

YOLOv8-Face models use anchor-free design with DFL (Distribution Focal Loss) for bbox regression. Provides detection with 5-point facial landmarks.

| Model Name       | Size   | Easy   | Medium | Hard   |
| ---------------- | ------ | ------ | ------ | ------ |
| `YOLOV8_LITE_S`| 7.4MB  | 93.4%  | 91.2%  | 78.6%  |
| `YOLOV8N` :material-check-circle: | 12MB   | 94.6%  | 92.3%  | 79.6%  |

!!! info "Accuracy & Benchmarks"
    **Accuracy**: WIDER FACE validation set (Easy/Medium/Hard subsets)

    **Speed**: Benchmark on your own hardware using `python tools/detect.py --source <image> --method yolov8face`

!!! note "Fixed Input Size"
    All YOLOv8-Face models use a fixed input size of 640×640.

---

## Face Recognition Models

### AdaFace

Face recognition using adaptive margin based on image quality.

| Model Name  | Backbone | Dataset     | Size   | IJB-B TAR | IJB-C TAR |
| ----------- | -------- | ----------- | ------ | --------- | --------- |
| `IR_18` :material-check-circle: | IR-18    | WebFace4M   | 92 MB  | 93.03%    | 94.99%    |
| `IR_101`  | IR-101   | WebFace12M  | 249 MB | -         | 97.66%    |

!!! info "Training Data & Accuracy"
    **Dataset**: [WebFace4M / WebFace12M](datasets.md#webface4m--webface12m) (4M / 12M images)

    **Accuracy**: IJB-B and IJB-C benchmarks, TAR@FAR=0.01%

!!! tip "Key Innovation"
    AdaFace introduces adaptive margin that adjusts based on image quality, providing better performance on low-quality images compared to fixed-margin approaches.


---

### ArcFace

Face recognition using additive angular margin loss.

| Model Name  | Backbone  | Params | Size  | LFW    | CFP-FP | AgeDB-30 | IJB-C |
| ----------- | --------- | ------ | ----- | ------ | ------ | -------- | ----- |
| `MNET` :material-check-circle: | MobileNet | 2.0M   | 8MB   | 99.70% | 98.00% | 96.58%   | 95.02% |
| `RESNET`  | ResNet50  | 43.6M  | 166MB | 99.83% | 99.33% | 98.23%   | 97.25% |

!!! info "Training Data"
    **Dataset**: Trained on [WebFace600K](datasets.md#webface600k) (600K images)

    **Accuracy**: IJB-C accuracy reported as TAR@FAR=1e-4

---

### MobileFace

Lightweight face recognition models with MobileNet backbones.

| Model Name        | Backbone         | Params | Size | LFW    | CALFW  | CPLFW  | AgeDB-30 |
| ----------------- | ---------------- | ------ | ---- | ------ | ------ | ------ | -------- |
| `MNET_025`      | MobileNetV1 0.25 | 0.36M  | 1MB  | 98.76% | 92.02% | 82.37% | 90.02%   |
| `MNET_V2` :material-check-circle:    | MobileNetV2      | 2.29M  | 4MB  | 99.55% | 94.87% | 86.89% | 95.16%   |
| `MNET_V3_SMALL` | MobileNetV3-S    | 1.25M  | 3MB  | 99.30% | 93.77% | 85.29% | 92.79%   |
| `MNET_V3_LARGE` | MobileNetV3-L    | 3.52M  | 10MB | 99.53% | 94.56% | 86.79% | 95.13%   |

!!! info "Training Data"
    **Dataset**: Trained on [MS1MV2](datasets.md#ms1mv2) (5.8M images, 85K identities)

    **Accuracy**: Evaluated on LFW, CALFW, CPLFW, and AgeDB-30 benchmarks

---

### SphereFace

Face recognition using angular softmax loss.

| Model Name   | Backbone | Params | Size | LFW    | CALFW  | CPLFW  | AgeDB-30 |
| ------------ | -------- | ------ | ---- | ------ | ------ | ------ | -------- |
| `SPHERE20` | Sphere20 | 24.5M  | 50MB | 99.67% | 95.61% | 88.75% | 96.58%   |
| `SPHERE36` | Sphere36 | 34.6M  | 92MB | 99.72% | 95.64% | 89.92% | 96.83%   |

!!! info "Training Data"
    **Dataset**: Trained on [MS1MV2](datasets.md#ms1mv2) (5.8M images, 85K identities)

    **Accuracy**: Evaluated on LFW, CALFW, CPLFW, and AgeDB-30 benchmarks

!!! note "Architecture"
    SphereFace uses angular softmax loss, an earlier approach before ArcFace. These models provide good accuracy with moderate resource requirements.

---

## Facial Landmark Models

### 106-Point Landmark Detection

Facial landmark localization model.

| Model Name | Points | Params | Size |
| ---------- | ------ | ------ | ---- |
| `2D106`  | 106    | 3.7M   | 14MB |

**Landmark Groups:**

| Group | Points | Count |
|-------|--------|-------|
| Face contour | 0-32 | 33 points |
| Eyebrows | 33-50 | 18 points |
| Nose | 51-62 | 12 points |
| Eyes | 63-86 | 24 points |
| Mouth | 87-105 | 19 points |

---

## Attribute Analysis Models

### Age & Gender Detection

| Model Name  | Attributes  | Params | Size |
| ----------- | ----------- | ------ | ---- |
| `AgeGender` | Age, Gender | 2.1M   | 8MB  |

!!! info "Training Data"
    **Dataset**: Trained on [CelebA](datasets.md#celeba)

!!! warning "Accuracy Note"
    Accuracy varies by demographic and image quality. Test on your specific use case.

---

### FairFace Attributes

| Model Name  | Attributes            | Params | Size  |
| ----------- | --------------------- | ------ | ----- |
| `FairFace` | Race, Gender, Age Group | -      | 44MB  |

!!! info "Training Data"
    **Dataset**: Trained on [FairFace](datasets.md#fairface) dataset with balanced demographics

!!! tip "Equitable Predictions"
    FairFace provides more equitable predictions across different racial and gender groups.

**Race Categories (7):** White, Black, Latino Hispanic, East Asian, Southeast Asian, Indian, Middle Eastern

**Age Groups (9):** 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+

---

### Emotion Detection

| Model Name    | Classes | Params | Size |
| ------------- | ------- | ------ | ---- |
| `AFFECNET7` | 7       | 0.5M   | 2MB  |
| `AFFECNET8` | 8       | 0.5M   | 2MB  |

**Classes (7)**: Neutral, Happy, Sad, Surprise, Fear, Disgust, Angry

**Classes (8)**: Above + Contempt

!!! info "Training Data"
    **Dataset**: Trained on [AffectNet](datasets.md#affectnet)

!!! note "Accuracy Note"
    Emotion detection accuracy depends heavily on facial expression clarity and cultural context.

---

## Gaze Estimation Models

### MobileGaze Family

Gaze direction prediction models trained on [Gaze360](datasets.md#gaze360) dataset. Returns pitch (vertical) and yaw (horizontal) angles in radians.

| Model Name     | Params | Size    | MAE*  |
| -------------- | ------ | ------- | ----- |
| `RESNET18`   | 11.7M  | 43 MB   | 12.84 |
| `RESNET34` :material-check-circle: | 24.8M  | 81.6 MB | 11.33 |
| `RESNET50`   | 25.6M  | 91.3 MB | 11.34 |
| `MOBILENET_V2` | 3.5M   | 9.59 MB | 13.07 |
| `MOBILEONE_S0` | 2.1M   | 4.8 MB  | 12.58 |

*MAE (Mean Absolute Error) in degrees on Gaze360 test set - lower is better

!!! info "Training Data"
    **Dataset**: Trained on [Gaze360](datasets.md#gaze360) (indoor/outdoor scenes with diverse head poses)

    **Training**: 200 epochs with classification-based approach (binned angles)

!!! note "Input Requirements"
    Requires face crop as input. Use face detection first to obtain bounding boxes.

---

## Face Parsing Models

### BiSeNet Family

BiSeNet (Bilateral Segmentation Network) models for semantic face parsing. Segments face images into 19 facial component classes.

| Model Name     | Params | Size    | Classes |
| -------------- | ------ | ------- | ------- |
| `RESNET18` :material-check-circle: | 13.3M  | 50.7 MB | 19      |
| `RESNET34`   | 24.1M  | 89.2 MB | 19      |

!!! info "Training Data"
    **Dataset**: Trained on [CelebAMask-HQ](datasets.md#celebamask-hq)

    **Architecture**: BiSeNet with ResNet backbone

    **Input Size**: 512×512 (automatically resized)

**19 Facial Component Classes:**

| # | Class | # | Class | # | Class |
|---|-------|---|-------|---|-------|
| 0 | Background | 7 | Left Ear | 14 | Neck |
| 1 | Skin | 8 | Right Ear | 15 | Neck Lace |
| 2 | Left Eyebrow | 9 | Ear Ring | 16 | Cloth |
| 3 | Right Eyebrow | 10 | Nose | 17 | Hair |
| 4 | Left Eye | 11 | Mouth | 18 | Hat |
| 5 | Right Eye | 12 | Upper Lip | | |
| 6 | Eye Glasses | 13 | Lower Lip | | |

**Applications:**

- Face makeup and beauty applications
- Virtual try-on systems
- Face editing and manipulation
- Facial feature extraction
- Portrait segmentation

!!! note "Input Requirements"
    Input should be a cropped face image. For full pipeline, use face detection first to obtain face crops.

---

### XSeg

XSeg from DeepFaceLab outputs masks for face regions. Requires 5-point landmarks for face alignment.

| Model Name | Size   | Output |
|------------|--------|--------|
| `DEFAULT`  | 67 MB  | Mask [0, 1] |

!!! info "Model Details"
    **Origin**: DeepFaceLab

    **Input**: NHWC format, normalized to [0, 1]

    **Alignment**: Requires 5-point landmarks (not bbox crops)

**Applications:**

- Face region extraction
- Face swapping pipelines
- Occlusion handling

!!! note "Input Requirements"
    Requires 5-point facial landmarks. Use a face detector like RetinaFace to obtain landmarks first.

---

## Anti-Spoofing Models

### MiniFASNet Family

Face anti-spoofing models for liveness detection. Detect if a face is real (live) or fake (photo, video replay, mask).

| Model Name | Size   | Scale |
| ---------- | ------ | ----- |
| `V1SE`   | 1.2 MB | 4.0   |
| `V2` :material-check-circle:  | 1.2 MB | 2.7   |

!!! info "Output Format"
    **Output**: Returns `SpoofingResult(is_real, confidence)` where is_real: True=Real, False=Fake

!!! note "Input Requirements"
    Requires face bounding box from a detector.

---

## Model Management

Models are automatically downloaded and cached on first use.

- **Cache location**: `~/.uniface/models/` (configurable via `set_cache_dir()` or `UNIFACE_CACHE_DIR` env var)
- **Inspect cache path**: `get_cache_dir()` returns the resolved active path
- **Verification**: Models are verified with SHA-256 checksums
- **Concurrent download**: `download_models([...])` fetches multiple models in parallel
- **Manual download**: Use `python tools/download_model.py` to pre-download models

See [Model Cache & Offline Use](concepts/model-cache-offline.md) for full details.

---

## References

### Model Training & Architectures

- **RetinaFace Training**: [yakhyo/retinaface-pytorch](https://github.com/yakhyo/retinaface-pytorch) - PyTorch implementation and training code
- **YOLOv5-Face Original**: [deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face) - Original PyTorch implementation
- **YOLOv5-Face ONNX**: [yakhyo/yolov5-face-onnx-inference](https://github.com/yakhyo/yolov5-face-onnx-inference) - ONNX inference implementation
- **YOLOv8-Face Original**: [derronqi/yolov8-face](https://github.com/derronqi/yolov8-face) - Original PyTorch implementation
- **YOLOv8-Face ONNX**: [yakhyo/yolov8-face-onnx-inference](https://github.com/yakhyo/yolov8-face-onnx-inference) - ONNX inference implementation
- **AdaFace Original**: [mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace) - Original PyTorch implementation
- **AdaFace ONNX**: [yakhyo/adaface-onnx](https://github.com/yakhyo/adaface-onnx) - ONNX export and inference
- **Face Recognition Training**: [yakhyo/face-recognition](https://github.com/yakhyo/face-recognition) - ArcFace, MobileFace, SphereFace training code
- **Gaze Estimation Training**: [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation) - MobileGaze training code and pretrained weights
- **Face Parsing Training**: [yakhyo/face-parsing](https://github.com/yakhyo/face-parsing) - BiSeNet training code and pretrained weights
- **Face Segmentation**: [yakhyo/face-segmentation](https://github.com/yakhyo/face-segmentation) - XSeg ONNX Inference
- **Face Anti-Spoofing**: [yakhyo/face-anti-spoofing](https://github.com/yakhyo/face-anti-spoofing) - MiniFASNet ONNX inference (weights from [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing))
- **FairFace**: [yakhyo/fairface-onnx](https://github.com/yakhyo/fairface-onnx) - FairFace ONNX inference for race, gender, age prediction
- **InsightFace**: [deepinsight/insightface](https://github.com/deepinsight/insightface) - Model architectures and pretrained weights

### Papers

- **RetinaFace**: [Single-Shot Multi-Level Face Localisation in the Wild](https://arxiv.org/abs/1905.00641)
- **SCRFD**: [Sample and Computation Redistribution for Efficient Face Detection](https://arxiv.org/abs/2105.04714)
- **YOLOv5-Face**: [YOLO5Face: Why Reinventing a Face Detector](https://arxiv.org/abs/2105.12931)
- **AdaFace**: [AdaFace: Quality Adaptive Margin for Face Recognition](https://arxiv.org/abs/2204.00964)
- **ArcFace**: [Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- **SphereFace**: [Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063)
- **BiSeNet**: [Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1808.00897)
