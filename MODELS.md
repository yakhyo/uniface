# UniFace Model Zoo

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
| `MNET_V2` ⭐ | 3.2M   | 3.5MB | 91.70% | 91.03% | 86.60% | **Recommended default** |
| `RESNET18`   | 11.7M  | 27MB  | 92.50% | 91.02% | 86.63% | Server/High accuracy          |
| `RESNET34`   | 24.8M  | 56MB  | 94.16% | 93.12% | 88.90% | Maximum accuracy              |

**Accuracy**: WIDER FACE validation set (Easy/Medium/Hard subsets) - from [RetinaFace paper](https://arxiv.org/abs/1905.00641)
**Speed**: Benchmark on your own hardware using `scripts/run_detection.py --iterations 100`

#### Usage

```python
from uniface import RetinaFace
from uniface.constants import RetinaFaceWeights

# Default (recommended)
detector = RetinaFace()  # Uses MNET_V2

# Specific model
detector = RetinaFace(
    model_name=RetinaFaceWeights.MNET_025,  # Fastest
    conf_thresh=0.5,
    nms_thresh=0.4,
    input_size=(640, 640)
)
```

---

### SCRFD Family

SCRFD (Sample and Computation Redistribution for Efficient Face Detection) models offer state-of-the-art speed-accuracy tradeoffs.

| Model Name       | Params | Size  | Easy   | Medium | Hard   | Use Case                        |
| ---------------- | ------ | ----- | ------ | ------ | ------ | ------------------------------- |
| `SCRFD_500M`   | 0.6M   | 2.5MB | 90.57% | 88.12% | 68.51% | Real-time applications          |
| `SCRFD_10G` ⭐ | 4.2M   | 17MB  | 95.16% | 93.87% | 83.05% | **High accuracy + speed** |

**Accuracy**: WIDER FACE validation set - from [SCRFD paper](https://arxiv.org/abs/2105.04714)
**Speed**: Benchmark on your own hardware using `scripts/run_detection.py --iterations 100`

#### Usage

```python
from uniface import SCRFD
from uniface.constants import SCRFDWeights

# Fast real-time detection
detector = SCRFD(
    model_name=SCRFDWeights.SCRFD_500M_KPS,
    conf_thresh=0.5,
    input_size=(640, 640)
)

# High accuracy
detector = SCRFD(
    model_name=SCRFDWeights.SCRFD_10G_KPS,
    conf_thresh=0.5
)
```

---

### YOLOv5-Face Family

YOLOv5-Face models provide excellent detection accuracy with 5-point facial landmarks, optimized for real-time applications.

| Model Name     | Size | Easy   | Medium | Hard   | Use Case                       |
| -------------- | ---- | ------ | ------ | ------ | ------------------------------ |
| `YOLOV5N`    | 11MB | 93.61% | 91.52% | 80.53% | Lightweight/Mobile             |
| `YOLOV5S` ⭐ | 28MB | 94.33% | 92.61% | 83.15% | **Real-time + accuracy** |
| `YOLOV5M`    | 82MB | 95.30% | 93.76% | 85.28% | High accuracy                  |

**Accuracy**: WIDER FACE validation set - from [YOLOv5-Face paper](https://arxiv.org/abs/2105.12931)
**Speed**: Benchmark on your own hardware using `scripts/run_detection.py --iterations 100`
**Note**: Fixed input size of 640×640. Models exported to ONNX from [deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face)

#### Usage

```python
from uniface import YOLOv5Face
from uniface.constants import YOLOv5FaceWeights

# Lightweight/Mobile
detector = YOLOv5Face(
    model_name=YOLOv5FaceWeights.YOLOV5N,
    conf_thresh=0.6,
    nms_thresh=0.5
)

# Real-time detection (recommended)
detector = YOLOv5Face(
    model_name=YOLOv5FaceWeights.YOLOV5S,
    conf_thresh=0.6,
    nms_thresh=0.5
)

# High accuracy
detector = YOLOv5Face(
    model_name=YOLOv5FaceWeights.YOLOV5M,
    conf_thresh=0.6
)

# Detect faces with landmarks
faces = detector.detect(image)
for face in faces:
    bbox = face['bbox']  # [x1, y1, x2, y2]
    confidence = face['confidence']
    landmarks = face['landmarks']  # 5-point landmarks (5, 2)
```

---

## Face Recognition Models

### ArcFace

State-of-the-art face recognition using additive angular margin loss.

| Model Name  | Backbone  | Params | Size  | Use Case                         |
| ----------- | --------- | ------ | ----- | -------------------------------- |
| `MNET` ⭐ | MobileNet | 2.0M   | 8MB   | **Balanced (recommended)** |
| `RESNET`  | ResNet50  | 43.6M  | 166MB | Maximum accuracy                 |

**Dataset**: Trained on MS1M-V2 (5.8M images, 85K identities)
**Accuracy**: Benchmark on your own dataset or use standard face verification benchmarks

#### Usage

```python
from uniface import ArcFace
from uniface.constants import ArcFaceWeights

# Default (MobileNet backbone)
recognizer = ArcFace()

# High accuracy (ResNet50 backbone)
recognizer = ArcFace(model_name=ArcFaceWeights.RESNET)

# Extract embedding
embedding = recognizer.get_normalized_embedding(image, landmarks)
# Returns: (1, 512) normalized embedding vector
```

---

### MobileFace

Lightweight face recognition optimized for mobile devices.

| Model Name        | Backbone         | Params | Size | LFW    | CALFW  | CPLFW  | AgeDB-30 | Use Case              |
| ----------------- | ---------------- | ------ | ---- | ------ | ------ | ------ | -------- | --------------------- |
| `MNET_025`      | MobileNetV1 0.25 | 0.36M  | 1MB  | 98.76% | 92.02% | 82.37% | 90.02%   | Ultra-lightweight     |
| `MNET_V2` ⭐    | MobileNetV2      | 2.29M  | 4MB  | 99.55% | 94.87% | 86.89% | 95.16%   | **Mobile/Edge** |
| `MNET_V3_SMALL` | MobileNetV3-S    | 1.25M  | 3MB  | 99.30% | 93.77% | 85.29% | 92.79%   | Mobile optimized      |
| `MNET_V3_LARGE` | MobileNetV3-L    | 3.52M  | 10MB | 99.53% | 94.56% | 86.79% | 95.13%   | Balanced mobile       |

**Dataset**: Trained on MS1M-V2 (5.8M images, 85K identities)
**Accuracy**: Evaluated on LFW, CALFW, CPLFW, and AgeDB-30 benchmarks
**Note**: These models are lightweight alternatives to ArcFace for resource-constrained environments

#### Usage

```python
from uniface import MobileFace
from uniface.constants import MobileFaceWeights

# Lightweight
recognizer = MobileFace(model_name=MobileFaceWeights.MNET_V2)
```

---

### SphereFace

Face recognition using angular softmax loss.

| Model Name   | Backbone | Params | Size | LFW    | CALFW  | CPLFW  | AgeDB-30 | Use Case            |
| ------------ | -------- | ------ | ---- | ------ | ------ | ------ | -------- | ------------------- |
| `SPHERE20` | Sphere20 | 24.5M  | 50MB | 99.67% | 95.61% | 88.75% | 96.58%   | Research/Comparison |
| `SPHERE36` | Sphere36 | 34.6M  | 92MB | 99.72% | 95.64% | 89.92% | 96.83%   | Research/Comparison |

**Dataset**: Trained on MS1M-V2 (5.8M images, 85K identities)
**Accuracy**: Evaluated on LFW, CALFW, CPLFW, and AgeDB-30 benchmarks
**Note**: SphereFace uses angular softmax loss, an earlier approach before ArcFace. These models provide good accuracy with moderate resource requirements.

#### Usage

```python
from uniface import SphereFace
from uniface.constants import SphereFaceWeights

recognizer = SphereFace(model_name=SphereFaceWeights.SPHERE20)
```

---

## Facial Landmark Models

### 106-Point Landmark Detection

High-precision facial landmark localization.

| Model Name | Points | Params | Size | Use Case                 |
| ---------- | ------ | ------ | ---- | ------------------------ |
| `2D106`  | 106    | 3.7M   | 14MB | Face alignment, analysis |

**Note**: Provides 106 facial keypoints for detailed face analysis and alignment

#### Usage

```python
from uniface import Landmark106

landmarker = Landmark106()
landmarks = landmarker.get_landmarks(image, bbox)
# Returns: (106, 2) array of (x, y) coordinates
```

**Landmark Groups:**

- Face contour: 0-32 (33 points)
- Eyebrows: 33-50 (18 points)
- Nose: 51-62 (12 points)
- Eyes: 63-86 (24 points)
- Mouth: 87-105 (19 points)

---

## Attribute Analysis Models

### Age & Gender Detection

| Model Name  | Attributes  | Params | Size | Use Case        |
| ----------- | ----------- | ------ | ---- | --------------- |
| `DEFAULT` | Age, Gender | 2.1M   | 8MB  | General purpose |

**Dataset**: Trained on CelebA
**Note**: Accuracy varies by demographic and image quality. Test on your specific use case.

#### Usage

```python
from uniface import AgeGender

predictor = AgeGender()
gender, age = predictor.predict(image, bbox)
# Returns: (gender, age_in_years)
# gender: 0 for Female, 1 for Male
```

---

### Emotion Detection

| Model Name    | Classes | Params | Size | Use Case        |
| ------------- | ------- | ------ | ---- | --------------- |
| `AFFECNET7` | 7       | 0.5M   | 2MB  | 7-class emotion |
| `AFFECNET8` | 8       | 0.5M   | 2MB  | 8-class emotion |

**Classes (7)**: Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger
**Classes (8)**: Above + Contempt

**Dataset**: Trained on AffectNet
**Note**: Emotion detection accuracy depends heavily on facial expression clarity and cultural context

#### Usage

```python
from uniface import Emotion
from uniface.constants import DDAMFNWeights

predictor = Emotion(model_name=DDAMFNWeights.AFFECNET7)
emotion, confidence = predictor.predict(image, landmarks)
```

---

## Gaze Estimation Models

### MobileGaze Family

Real-time gaze direction prediction models trained on Gaze360 dataset. Returns pitch (vertical) and yaw (horizontal) angles in radians.

| Model Name     | Params | Size    | MAE*  | Use Case                      |
| -------------- | ------ | ------- | ----- | ----------------------------- |
| `RESNET18`   | 11.7M  | 43 MB   | 12.84 | Balanced accuracy/speed       |
| `RESNET34` ⭐ | 24.8M  | 81.6 MB | 11.33 | **Recommended default** |
| `RESNET50`   | 25.6M  | 91.3 MB | 11.34 | High accuracy                 |
| `MOBILENET_V2` | 3.5M   | 9.59 MB | 13.07 | Mobile/Edge devices           |
| `MOBILEONE_S0` | 2.1M   | 4.8 MB  | 12.58 | Lightweight/Real-time         |

*MAE (Mean Absolute Error) in degrees on Gaze360 test set - lower is better

**Dataset**: Trained on Gaze360 (indoor/outdoor scenes with diverse head poses)
**Training**: 200 epochs with classification-based approach (binned angles)

#### Usage

```python
from uniface import MobileGaze
from uniface.constants import GazeWeights
import numpy as np

# Default (recommended)
gaze_estimator = MobileGaze()  # Uses RESNET34

# Lightweight model
gaze_estimator = MobileGaze(model_name=GazeWeights.MOBILEONE_S0)

# Estimate gaze from face crop
pitch, yaw = gaze_estimator.estimate(face_crop)
print(f"Pitch: {np.degrees(pitch):.1f}°, Yaw: {np.degrees(yaw):.1f}°")
```

**Note**: Requires face crop as input. Use face detection first to obtain bounding boxes.

---

## Model Updates

Models are automatically downloaded and cached on first use. Cache location: `~/.uniface/models/`

### Manual Model Management

```python
from uniface.model_store import verify_model_weights
from uniface.constants import RetinaFaceWeights

# Download specific model
model_path = verify_model_weights(
    RetinaFaceWeights.MNET_V2,
    root='./custom_cache'
)

# Models are verified with SHA-256 checksums
```

### Download All Models

```bash
# Using the provided script
python scripts/download_model.py

# Download specific model
python scripts/download_model.py --model MNET_V2
```

---

## References

### Model Training & Architectures

- **RetinaFace Training**: [yakhyo/retinaface-pytorch](https://github.com/yakhyo/retinaface-pytorch) - PyTorch implementation and training code
- **YOLOv5-Face Original**: [deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face) - Original PyTorch implementation
- **YOLOv5-Face ONNX**: [yakhyo/yolov5-face-onnx-inference](https://github.com/yakhyo/yolov5-face-onnx-inference) - ONNX inference implementation
- **Face Recognition Training**: [yakhyo/face-recognition](https://github.com/yakhyo/face-recognition) - ArcFace, MobileFace, SphereFace training code
- **Gaze Estimation Training**: [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation) - MobileGaze training code and pretrained weights
- **InsightFace**: [deepinsight/insightface](https://github.com/deepinsight/insightface) - Model architectures and pretrained weights

### Papers

- **RetinaFace**: [Single-Shot Multi-Level Face Localisation in the Wild](https://arxiv.org/abs/1905.00641)
- **SCRFD**: [Sample and Computation Redistribution for Efficient Face Detection](https://arxiv.org/abs/2105.04714)
- **YOLOv5-Face**: [YOLO5Face: Why Reinventing a Face Detector](https://arxiv.org/abs/2105.12931)
- **ArcFace**: [Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- **SphereFace**: [Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063)
