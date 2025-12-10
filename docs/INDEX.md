# MLX-UniFace Documentation

Welcome to the MLX-UniFace documentation. This library provides production-ready face analysis with dual backend support for Apple Silicon (MLX) and cross-platform (ONNX).

---

## Documentation Structure

| Document | Description |
|----------|-------------|
| [API Reference](API_REFERENCE.md) | Complete API documentation with all classes, methods, and parameters |
| [Architecture](ARCHITECTURE.md) | System architecture, design patterns, and component diagrams |
| [User Guide](USER_GUIDE.md) | Practical tutorials and usage examples |

---

## Quick Links

### Getting Started
- [Installation](USER_GUIDE.md#installation)
- [Quick Start](USER_GUIDE.md#quick-start)

### Core Features
- [Face Detection](API_REFERENCE.md#detection)
- [Face Recognition](API_REFERENCE.md#recognition)
- [Age & Gender](API_REFERENCE.md#attributes)
- [Landmarks](API_REFERENCE.md#landmarks)

### Advanced Topics
- [Backend Configuration](API_REFERENCE.md#backend-configuration)
- [Architecture Overview](ARCHITECTURE.md#high-level-overview)
- [Performance Optimization](USER_GUIDE.md#performance-optimization)

---

## Overview

### What is MLX-UniFace?

MLX-UniFace is a blazing-fast face analysis library optimized for Apple Silicon. It provides:

- **Face Detection**: RetinaFace, SCRFD, YOLOv5Face
- **Face Recognition**: ArcFace, MobileFace, SphereFace (512-dim embeddings)
- **Face Attributes**: Age, gender, emotion prediction
- **Facial Landmarks**: 5-point and 106-point detection

### Key Features

| Feature | Description |
|---------|-------------|
| Dual Backend | MLX (Apple Silicon) + ONNX (cross-platform) |
| Auto-Selection | Automatically uses best available backend |
| Model Verification | SHA-256 hash verification for all downloads |
| Production Ready | Comprehensive error handling and logging |

### Installation

```bash
# Apple Silicon (recommended)
pip install mlx-uniface[mlx]

# Cross-platform
pip install mlx-uniface[onnx]

# All backends
pip install mlx-uniface[all]
```

### Minimal Example

```python
from uniface import detect_faces, draw_detections
import cv2

image = cv2.imread('photo.jpg')
faces = detect_faces(image)
result = draw_detections(image, faces)
cv2.imwrite('result.jpg', result)
```

---

## Available Models

### Detection Models

| Model | Variants | Paper |
|-------|----------|-------|
| RetinaFace | MNET_025, MNET_050, MNET_V1, MNET_V2, RESNET18, RESNET34 | [arXiv](https://arxiv.org/abs/1905.00641) |
| SCRFD | SCRFD_10G_KPS, SCRFD_500M_KPS | [arXiv](https://arxiv.org/abs/2105.04714) |
| YOLOv5Face | YOLOV5S, YOLOV5M | [arXiv](https://arxiv.org/abs/2105.12931) |

### Recognition Models

| Model | Variants | Training Data |
|-------|----------|---------------|
| ArcFace | MNET, RESNET | MS1M V2 (5.8M images) |
| MobileFace | MNET_025, MNET_V2, MNET_V3_SMALL, MNET_V3_LARGE | MS1M V2 |
| SphereFace | SPHERE20, SPHERE36 | MS1M V2 |

---

## Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/CodeWithBehnam/mlx-uniface/issues)
- **Original UniFace**: [yakhyo/uniface](https://github.com/yakhyo/uniface)

---

*Documentation for MLX-UniFace v1.3.1*
