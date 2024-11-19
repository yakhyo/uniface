# RetinaFace: Lightweight Face Detection Library

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
[![Build Status](https://github.com/yakhyo/retinaface/actions/workflows/build.yml/badge.svg)](https://github.com/yakhyo/retinaface/actions)
[![Downloads](https://pepy.tech/badge/retinaface)](https://pepy.tech/project/retinaface)
[![Code Style: PEP8](https://img.shields.io/badge/code%20style-PEP8-green.svg)](https://www.python.org/dev/peps/pep-0008/)
[![GitHub Stars](https://img.shields.io/github/stars/yakhyo/retinaface.svg)](https://github.com/yakhyo/retinaface/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/yakhyo/retinaface.svg)](https://github.com/yakhyo/retinaface/network)
[![Issues](https://img.shields.io/github/issues/yakhyo/retinaface.svg)](https://github.com/yakhyo/retinaface/issues)

RetinaFace is a lightweight face detection library designed for high-performance face localization and landmark detection. The library supports ONNX models and provides utilities for bounding box visualization and landmark plotting.

---

## Features

- High-speed face detection using ONNX models.
- Accurate facial landmark localization (e.g., eyes, nose, and mouth).
- Easy-to-use API for inference and visualization.
- Customizable confidence thresholds for bounding box filtering.

---

## Installation

### Using pip

```bash
pip install retinaface
```

### Clone the Repository

```bash
git clone https://github.com/yourusername/retinaface.git
cd retinaface
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Initialize the Model

```python
from retinaface import RetinaFace

# Initialize the RetinaFace model
retinaface_inference = RetinaFace(
    model="retinaface_mnet_v2",  # Model name
    conf_thresh=0.5,            # Confidence threshold
    pre_nms_topk=5000,          # Pre-NMS Top-K detections
    nms_thresh=0.4,             # NMS IoU threshold
    post_nms_topk=750           # Post-NMS Top-K detections
)
```

### Run Inference

```python
import cv2
from retinaface.visualization import draw_detections

# Load an image
image_path = "assets/test.jpg"
original_image = cv2.imread(image_path)

# Perform inference
boxes, landmarks = retinaface_inference.detect(original_image)

# Visualize results
draw_detections(original_image, (boxes, landmarks), vis_threshold=0.6)

# Save the output image
output_path = "output.jpg"
cv2.imwrite(output_path, original_image)
print(f"Saved output image to {output_path}")
```

---

## API Reference

### RetinaFace Class

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

- `model`: Model name (e.g., retinaface_mnet_v2).
  - retinaface_mnet025
  - retinaface_mnet050
  - retinaface_mnet_v1
  - retinaface_mnet_v2
  - retinaface_r18
  - retinaface_r34
- `conf_thresh`: Minimum confidence threshold for detections.
- `pre_nms_topk`: Maximum number of detections to keep before NMS.
- `nms_thresh`: IoU threshold for Non-Maximum Suppression.
- `post_nms_topk`: Maximum number of detections to keep after NMS.

#### `detect(image: np.ndarray, max_num: Optional[int] = 0, metric: Literal["default", "max"] = "default", center_weight: Optional[float] = 2.0) -> Tuple[np.ndarray, np.ndarray]`

- **Description**: Performs face detection on the input image and returns bounding boxes and landmarks for detected faces.

- **Inputs**:

  - `image` (`np.ndarray`): The input image as a NumPy array in BGR format.
  - `max_num` (`Optional[int]`, default=`0`): The maximum number of faces to return. If `0`, all detected faces are returned.
  - `metric` (`Literal["default", "max"]`, default=`"default"`): The metric for prioritizing detections:
    - `"default"`: Prioritize detections closer to the image center.
    - `"max"`: Prioritize detections with larger bounding box areas.
  - `center_weight` (`Optional[float]`, default=`2.0`): A weight factor for prioritizing faces closer to the center of the image.

- **Outputs**:
  - `Tuple[np.ndarray, np.ndarray]`: A tuple containing:
    - `bounding_boxes` (`np.ndarray`): An array of bounding boxes, each represented as `[x_min, y_min, x_max, y_max, confidence]`.
    - `landmarks` (`np.ndarray`): An array of facial landmarks, each represented as `[(x1, y1), ..., (x5, y5)]`.

---

## Visualization Utilities

### `draw_detections(original_image, detections, vis_threshold)`

- Draws bounding boxes and landmarks on the image.
- Filters detections below the confidence threshold.

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

- Based on the RetinaFace model for face detection.
- Inspired by InsightFace and other face detection projects.

---
