# UniFace: All-in-One Face Analysis Library

<div align="center">

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
[![PyPI Version](https://img.shields.io/pypi/v/uniface.svg)](https://pypi.org/project/uniface/)
[![Build Status](https://github.com/yakhyo/uniface/actions/workflows/build.yml/badge.svg)](https://github.com/yakhyo/uniface/actions)
[![Downloads](https://pepy.tech/badge/uniface)](https://pepy.tech/project/uniface)
[![Code Style: PEP8](https://img.shields.io/badge/code%20style-PEP8-green.svg)](https://www.python.org/dev/peps/pep-0008/)
[![GitHub Release Downloads](https://img.shields.io/github/downloads/yakhyo/uniface/total.svg?label=Model%20Downloads)](https://github.com/yakhyo/uniface/releases)

</div>

**uniface** is a lightweight face detection library designed for high-performance face localization and landmark detection. The library supports ONNX models and provides utilities for bounding box visualization and landmark plotting. To train RetinaFace model, see https://github.com/yakhyo/retinaface-pytorch.

---

## Features
- [ ] Age and gender detection (Planned).
- [ ] Face recognition (Planned).
- [x] High-speed face detection using ONNX models (Added: 2024-11-20).
- [x] Accurate facial landmark localization (e.g., eyes, nose, and mouth) (Added: 2024-11-20).
- [x] Easy-to-use API for inference and visualization (Added: 2024-11-20).

---

## Installation

### Using pip

```bash
pip install uniface
```

### Local installation using pip

**Clone the repository**

```bash
git clone https://github.com/yakhyo/uniface.git
cd uniface
```

**Install using pip**

```bash
pip install .
```

---

## Quick Start

### Initialize the Model

```python
from uniface import RetinaFace

# Initialize the RetinaFace model
uniface_inference = RetinaFace(
    model="retinaface_mnet_v2",  # Model name
    conf_thresh=0.5,             # Confidence threshold
    pre_nms_topk=5000,           # Pre-NMS Top-K detections
    nms_thresh=0.4,              # NMS IoU threshold
    post_nms_topk=750            # Post-NMS Top-K detections
)
```

### Run Inference

Inference on image:

```python
import cv2
from uniface.visualization import draw_detections

# Load an image
image_path = "assets/test.jpg"
original_image = cv2.imread(image_path)

# Perform inference
boxes, landmarks = uniface_inference.detect(original_image)

# Visualize results
draw_detections(original_image, (boxes, landmarks), vis_threshold=0.6)

# Save the output image
output_path = "output.jpg"
cv2.imwrite(output_path, original_image)
print(f"Saved output image to {output_path}")
```

Inference on video:

```python
import cv2
from uniface.visualization import draw_detections

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    # Perform inference
    boxes, landmarks = uniface_inference.detect(frame)

    # Draw detections on the frame
    draw_detections(frame, (boxes, landmarks), vis_threshold=0.6)

    # Display the output
    cv2.imshow("Webcam Inference", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
```

---

### Evaluation results of available models on WiderFace

| RetinaFace Models  | Easy       | Medium     | Hard       |
| ------------------ | ---------- | ---------- | ---------- |
| retinaface_mnet025 | 88.48%     | 87.02%     | 80.61%     |
| retinaface_mnet050 | 89.42%     | 87.97%     | 82.40%     |
| retinaface_mnet_v1 | 90.59%     | 89.14%     | 84.13%     |
| retinaface_mnet_v2 | 91.70%     | 91.03%     | 86.60%     |
| retinaface_r18     | 92.50%     | 91.02%     | 86.63%     |
| retinaface_r34     | **94.16%** | **93.12%** | **88.90%** |

## API Reference

### `RetinaFace` Class

#### Initialization
```python
RetinaFace(
    model: str,
    conf_thresh: float = 0.5,
    pre_nms_topk: int = 5000,
    nms_thresh: float = 0.4,
    post_nms_topk: int = 750
)
```

**Parameters**:
- `model` *(str)*: Name of the model to use. Supported models:
  - `retinaface_mnet025`, `retinaface_mnet050`, `retinaface_mnet_v1`, `retinaface_mnet_v2`
  - `retinaface_r18`, `retinaface_r34`
- `conf_thresh` *(float, default=0.5)*: Minimum confidence score for detections.
- `pre_nms_topk` *(int, default=5000)*: Max detections to keep before NMS.
- `nms_thresh` *(float, default=0.4)*: IoU threshold for Non-Maximum Suppression.
- `post_nms_topk` *(int, default=750)*: Max detections to keep after NMS.

---

### `detect` Method
```python
detect(
    image: np.ndarray,
    max_num: int = 0,
    metric: str = "default",
    center_weight: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]
```

**Description**:
Detects faces in the given image and returns bounding boxes and landmarks.

**Parameters**:
- `image` *(np.ndarray)*: Input image in BGR format.
- `max_num` *(int, default=0)*: Maximum number of faces to return. `0` means return all.
- `metric` *(str, default="default")*: Metric for prioritizing detections:
  - `"default"`: Prioritize detections closer to the image center.
  - `"max"`: Prioritize larger bounding box areas.
- `center_weight` *(float, default=2.0)*: Weight for prioritizing center-aligned faces.

**Returns**:
- `bounding_boxes` *(np.ndarray)*: Array of detections as `[x_min, y_min, x_max, y_max, confidence]`.
- `landmarks` *(np.ndarray)*: Array of landmarks as `[(x1, y1), ..., (x5, y5)]`.

---

### Visualization Utilities

#### `draw_detections`
```python
draw_detections(
    image: np.ndarray,
    detections: Tuple[np.ndarray, np.ndarray],
    vis_threshold: float
) -> None
```

**Description**:
Draws bounding boxes and landmarks on the given image.

**Parameters**:
- `image` *(np.ndarray)*: The input image in BGR format.
- `detections` *(Tuple[np.ndarray, np.ndarray])*: A tuple of bounding boxes and landmarks.
- `vis_threshold` *(float)*: Minimum confidence score for visualization.

---

## Contributing

We welcome contributions to enhance the library! Feel free to:

- Submit bug reports or feature requests.
- Fork the repository and create a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Based on the RetinaFace model for face detection ([https://github.com/yakhyo/retinaface-pytorch](https://github.com/yakhyo/retinaface-pytorch)).
- Inspired by InsightFace and other face detection projects.

---
