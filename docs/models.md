# Model Zoo

Complete guide to all available models, their performance characteristics, and selection criteria.

---

## Face Detection Models

### RetinaFace Family

RetinaFace models are trained on the WIDER FACE dataset and provide excellent accuracy-speed tradeoffs.

| Model Name     | Params | Size  | Easy   | Medium | Hard   | Use Case                      |
| -------------- | ------ | ----- | ------ | ------ | ------ | ----------------------------- |
| `MNET_025`   | 0.4M   | 1.7MB | 88.48% | 87.02% | 80.61% | Mobile/Edge devices           |
| `MNET_050`   | 1.0M   | 2.6MB | 89.42% | 87.97% | 82.40% | Mobile/Edge devices           |
| `MNET_V1`    | 3.5M   | 3.8MB | 90.59% | 89.14% | 84.13% | Balanced mobile               |
| `MNET_V2` :material-check-circle: | 3.2M   | 3.5MB | 91.70% | 91.03% | 86.60% | **Default** |
| `RESNET18`   | 11.7M  | 27MB  | 92.50% | 91.02% | 86.63% | Server/High accuracy          |
| `RESNET34`   | 24.8M  | 56MB  | 94.16% | 93.12% | 88.90% | Maximum accuracy              |

!!! info "Accuracy & Benchmarks"
    **Accuracy**: WIDER FACE validation set (Easy/Medium/Hard subsets) - from [RetinaFace paper](https://arxiv.org/abs/1905.00641)

    **Speed**: Benchmark on your own hardware using `python tools/detection.py --source <image> --iterations 100`

---

### SCRFD Family

SCRFD (Sample and Computation Redistribution for Efficient Face Detection) models offer state-of-the-art speed-accuracy tradeoffs.

| Model Name       | Params | Size  | Easy   | Medium | Hard   | Use Case                        |
| ---------------- | ------ | ----- | ------ | ------ | ------ | ------------------------------- |
| `SCRFD_500M`   | 0.6M   | 2.5MB | 90.57% | 88.12% | 68.51% | Real-time applications          |
| `SCRFD_10G` :material-check-circle: | 4.2M   | 17MB  | 95.16% | 93.87% | 83.05% | **High accuracy + speed** |

!!! info "Accuracy & Benchmarks"
    **Accuracy**: WIDER FACE validation set - from [SCRFD paper](https://arxiv.org/abs/2105.04714)

    **Speed**: Benchmark on your own hardware using `python tools/detection.py --source <image> --iterations 100`

---

### YOLOv5-Face Family

YOLOv5-Face models provide excellent detection accuracy with 5-point facial landmarks, optimized for real-time applications.

| Model Name     | Size | Easy   | Medium | Hard   | Use Case                       |
| -------------- | ---- | ------ | ------ | ------ | ------------------------------ |
| `YOLOV5N`    | 11MB | 93.61% | 91.52% | 80.53% | Lightweight/Mobile             |
| `YOLOV5S` :material-check-circle: | 28MB | 94.33% | 92.61% | 83.15% | **Real-time + accuracy** |
| `YOLOV5M`    | 82MB | 95.30% | 93.76% | 85.28% | High accuracy                  |

!!! info "Accuracy & Benchmarks"
    **Accuracy**: WIDER FACE validation set - from [YOLOv5-Face paper](https://arxiv.org/abs/2105.12931)

    **Speed**: Benchmark on your own hardware using `python tools/detection.py --source <image> --iterations 100`

!!! note "Fixed Input Size"
    All YOLOv5-Face models use a fixed input size of 640×640. Models exported to ONNX from [deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face).

---

## Face Recognition Models

### AdaFace

High-quality face recognition using adaptive margin based on image quality. Achieves state-of-the-art results on challenging benchmarks.

| Model Name  | Backbone | Dataset     | Size   | IJB-B TAR | IJB-C TAR | Use Case              |
| ----------- | -------- | ----------- | ------ | --------- | --------- | --------------------- |
| `IR_18` :material-check-circle: | IR-18    | WebFace4M   | 92 MB  | 93.03%    | 94.99%    | **Balanced (default)** |
| `IR_101`  | IR-101   | WebFace12M  | 249 MB | -         | 97.66%    | Maximum accuracy       |

!!! info "Training Data & Accuracy"
    **Dataset**: WebFace4M (4M images) / WebFace12M (12M images)

    **Accuracy**: IJB-B and IJB-C benchmarks, TAR@FAR=0.01%

!!! tip "Key Innovation"
    AdaFace introduces adaptive margin that adjusts based on image quality, providing better performance on low-quality images compared to fixed-margin approaches.

**Reference**: [AdaFace: Quality Adaptive Margin for Face Recognition](https://github.com/mk-minchul/AdaFace) | [ONNX Export](https://github.com/yakhyo/adaface-onnx)

---

### ArcFace

State-of-the-art face recognition using additive angular margin loss.

| Model Name  | Backbone  | Params | Size  | Use Case                         |
| ----------- | --------- | ------ | ----- | -------------------------------- |
| `MNET` :material-check-circle: | MobileNet | 2.0M   | 8MB   | **Balanced (recommended)** |
| `RESNET`  | ResNet50  | 43.6M  | 166MB | Maximum accuracy                 |

!!! info "Training Data"
    **Dataset**: Trained on MS1M-V2 (5.8M images, 85K identities)

    **Accuracy**: Benchmark on your own dataset or use standard face verification benchmarks

---

### MobileFace

Lightweight face recognition optimized for mobile devices.

| Model Name        | Backbone         | Params | Size | LFW    | CALFW  | CPLFW  | AgeDB-30 | Use Case              |
| ----------------- | ---------------- | ------ | ---- | ------ | ------ | ------ | -------- | --------------------- |
| `MNET_025`      | MobileNetV1 0.25 | 0.36M  | 1MB  | 98.76% | 92.02% | 82.37% | 90.02%   | Ultra-lightweight     |
| `MNET_V2` :material-check-circle:    | MobileNetV2      | 2.29M  | 4MB  | 99.55% | 94.87% | 86.89% | 95.16%   | **Mobile/Edge** |
| `MNET_V3_SMALL` | MobileNetV3-S    | 1.25M  | 3MB  | 99.30% | 93.77% | 85.29% | 92.79%   | Mobile optimized      |
| `MNET_V3_LARGE` | MobileNetV3-L    | 3.52M  | 10MB | 99.53% | 94.56% | 86.79% | 95.13%   | Balanced mobile       |

!!! info "Training Data"
    **Dataset**: Trained on MS1M-V2 (5.8M images, 85K identities)

    **Accuracy**: Evaluated on LFW, CALFW, CPLFW, and AgeDB-30 benchmarks

!!! tip "Use Case"
    These models are lightweight alternatives to ArcFace for resource-constrained environments.

---

### SphereFace

Face recognition using angular softmax loss.

| Model Name   | Backbone | Params | Size | LFW    | CALFW  | CPLFW  | AgeDB-30 | Use Case            |
| ------------ | -------- | ------ | ---- | ------ | ------ | ------ | -------- | ------------------- |
| `SPHERE20` | Sphere20 | 24.5M  | 50MB | 99.67% | 95.61% | 88.75% | 96.58%   | Research/Comparison |
| `SPHERE36` | Sphere36 | 34.6M  | 92MB | 99.72% | 95.64% | 89.92% | 96.83%   | Research/Comparison |

!!! info "Training Data"
    **Dataset**: Trained on MS1M-V2 (5.8M images, 85K identities)

    **Accuracy**: Evaluated on LFW, CALFW, CPLFW, and AgeDB-30 benchmarks

!!! note "Architecture"
    SphereFace uses angular softmax loss, an earlier approach before ArcFace. These models provide good accuracy with moderate resource requirements.

---

## Facial Landmark Models

### 106-Point Landmark Detection

High-precision facial landmark localization.

| Model Name | Points | Params | Size | Use Case                 |
| ---------- | ------ | ------ | ---- | ------------------------ |
| `2D106`  | 106    | 3.7M   | 14MB | Face alignment, analysis |

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

| Model Name  | Attributes  | Params | Size | Use Case        |
| ----------- | ----------- | ------ | ---- | --------------- |
| `AgeGender` | Age, Gender | 2.1M   | 8MB  | General purpose |

!!! info "Training Data"
    **Dataset**: Trained on CelebA

!!! warning "Accuracy Note"
    Accuracy varies by demographic and image quality. Test on your specific use case.

---

### FairFace Attributes

| Model Name  | Attributes            | Params | Size  | Use Case                    |
| ----------- | --------------------- | ------ | ----- | --------------------------- |
| `FairFace` | Race, Gender, Age Group | -      | 44MB  | Balanced demographic prediction |

!!! info "Training Data"
    **Dataset**: Trained on FairFace dataset with balanced demographics

!!! tip "Equitable Predictions"
    FairFace provides more equitable predictions across different racial and gender groups.

**Race Categories (7):** White, Black, Latino Hispanic, East Asian, Southeast Asian, Indian, Middle Eastern

**Age Groups (9):** 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+

---

### Emotion Detection

| Model Name    | Classes | Params | Size | Use Case        |
| ------------- | ------- | ------ | ---- | --------------- |
| `AFFECNET7` | 7       | 0.5M   | 2MB  | 7-class emotion |
| `AFFECNET8` | 8       | 0.5M   | 2MB  | 8-class emotion |

**Classes (7)**: Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger

**Classes (8)**: Above + Contempt

!!! info "Training Data"
    **Dataset**: Trained on AffectNet

!!! note "Accuracy Note"
    Emotion detection accuracy depends heavily on facial expression clarity and cultural context.

---

## Gaze Estimation Models

### MobileGaze Family

Real-time gaze direction prediction models trained on Gaze360 dataset. Returns pitch (vertical) and yaw (horizontal) angles in radians.

| Model Name     | Params | Size    | MAE*  | Use Case                      |
| -------------- | ------ | ------- | ----- | ----------------------------- |
| `RESNET18`   | 11.7M  | 43 MB   | 12.84 | Balanced accuracy/speed       |
| `RESNET34` :material-check-circle: | 24.8M  | 81.6 MB | 11.33 | **Default** |
| `RESNET50`   | 25.6M  | 91.3 MB | 11.34 | High accuracy                 |
| `MOBILENET_V2` | 3.5M   | 9.59 MB | 13.07 | Mobile/Edge devices           |
| `MOBILEONE_S0` | 2.1M   | 4.8 MB  | 12.58 | Lightweight/Real-time         |

*MAE (Mean Absolute Error) in degrees on Gaze360 test set - lower is better

!!! info "Training Data"
    **Dataset**: Trained on Gaze360 (indoor/outdoor scenes with diverse head poses)

    **Training**: 200 epochs with classification-based approach (binned angles)

!!! note "Input Requirements"
    Requires face crop as input. Use face detection first to obtain bounding boxes.

---

## Face Parsing Models

### BiSeNet Family

BiSeNet (Bilateral Segmentation Network) models for semantic face parsing. Segments face images into 19 facial component classes.

| Model Name     | Params | Size    | Classes | Use Case                      |
| -------------- | ------ | ------- | ------- | ----------------------------- |
| `RESNET18` :material-check-circle: | 13.3M  | 50.7 MB | 19      | **Default** |
| `RESNET34`   | 24.1M  | 89.2 MB | 19      | Higher accuracy               |

!!! info "Training Data"
    **Dataset**: Trained on CelebAMask-HQ

    **Architecture**: BiSeNet with ResNet backbone

    **Input Size**: 512×512 (automatically resized)

**19 Facial Component Classes:**

| # | Class | # | Class | # | Class |
|---|-------|---|-------|---|-------|
| 1 | Background | 8 | Left Ear | 15 | Neck |
| 2 | Skin | 9 | Right Ear | 16 | Neck Lace |
| 3 | Left Eyebrow | 10 | Ear Ring | 17 | Cloth |
| 4 | Right Eyebrow | 11 | Nose | 18 | Hair |
| 5 | Left Eye | 12 | Mouth | 19 | Hat |
| 6 | Right Eye | 13 | Upper Lip | | |
| 7 | Eye Glasses | 14 | Lower Lip | | |

**Applications:**

- Face makeup and beauty applications
- Virtual try-on systems
- Face editing and manipulation
- Facial feature extraction
- Portrait segmentation

!!! note "Input Requirements"
    Input should be a cropped face image. For full pipeline, use face detection first to obtain face crops.

---

## Anti-Spoofing Models

### MiniFASNet Family

Lightweight face anti-spoofing models for liveness detection. Detect if a face is real (live) or fake (photo, video replay, mask).

| Model Name | Size   | Scale | Use Case                      |
| ---------- | ------ | ----- | ----------------------------- |
| `V1SE`   | 1.2 MB | 4.0   | Squeeze-and-excitation variant |
| `V2` :material-check-circle:  | 1.2 MB | 2.7   | **Default**       |

!!! info "Output Format"
    **Output**: Returns `SpoofingResult(is_real, confidence)` where is_real: True=Real, False=Fake

!!! note "Input Requirements"
    Requires face bounding box from a detector. Use with RetinaFace, SCRFD, or YOLOv5Face.

---

## Model Management

Models are automatically downloaded and cached on first use.

- **Cache location**: `~/.uniface/models/`
- **Verification**: Models are verified with SHA-256 checksums
- **Manual download**: Use `python tools/download_model.py` to pre-download models

---

## References

### Model Training & Architectures

- **RetinaFace Training**: [yakhyo/retinaface-pytorch](https://github.com/yakhyo/retinaface-pytorch) - PyTorch implementation and training code
- **YOLOv5-Face Original**: [deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face) - Original PyTorch implementation
- **YOLOv5-Face ONNX**: [yakhyo/yolov5-face-onnx-inference](https://github.com/yakhyo/yolov5-face-onnx-inference) - ONNX inference implementation
- **AdaFace Original**: [mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace) - Original PyTorch implementation
- **AdaFace ONNX**: [yakhyo/adaface-onnx](https://github.com/yakhyo/adaface-onnx) - ONNX export and inference
- **Face Recognition Training**: [yakhyo/face-recognition](https://github.com/yakhyo/face-recognition) - ArcFace, MobileFace, SphereFace training code
- **Gaze Estimation Training**: [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation) - MobileGaze training code and pretrained weights
- **Face Parsing Training**: [yakhyo/face-parsing](https://github.com/yakhyo/face-parsing) - BiSeNet training code and pretrained weights
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
