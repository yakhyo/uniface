# Custom Models

Add your own ONNX models to UniFace.

!!! note "Work in Progress"
    This page contains example code patterns for advanced users. Test thoroughly before using in production.

---

## Overview

UniFace is designed to be extensible. You can add custom ONNX models by:

1. Creating a class that inherits from the appropriate base class
2. Implementing required methods
3. Using the ONNX Runtime utilities provided by UniFace

---

## Add Custom Detection Model

```python
from uniface.detection.base import BaseDetector
from uniface.onnx_utils import create_onnx_session
from uniface.types import Face
import numpy as np

class MyDetector(BaseDetector):
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        super().__init__(confidence_threshold=confidence_threshold)
        self.session = create_onnx_session(model_path)
        self.threshold = confidence_threshold

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # Your preprocessing logic
        # e.g., resize, normalize, transpose
        raise NotImplementedError

    def postprocess(self, outputs, shape) -> list[Face]:
        # Your postprocessing logic
        # e.g., decode boxes, apply NMS, create Face objects
        raise NotImplementedError

    def detect(self, image: np.ndarray) -> list[Face]:
        # 1. Preprocess image
        input_tensor = self.preprocess(image)

        # 2. Run inference
        outputs = self.session.run(None, {'input': input_tensor})

        # 3. Postprocess outputs to Face objects
        return self.postprocess(outputs, image.shape)
```

---

## Add Custom Recognition Model

```python
from uniface.recognition.base import BaseRecognizer, PreprocessConfig

class MyRecognizer(BaseRecognizer):
    def __init__(self, model_path: str, providers=None):
        preprocessing = PreprocessConfig(input_mean=127.5, input_std=127.5, input_size=(112, 112))
        super().__init__(model_path, preprocessing, providers=providers)

    # Optional: override preprocess() if your model expects custom normalization.
```

---

## Usage

```python
from my_module import MyDetector, MyRecognizer

# Use custom models
detector = MyDetector("path/to/detection_model.onnx")
recognizer = MyRecognizer("path/to/recognition_model.onnx")

# Use like built-in models
faces = detector.detect(image)
embedding = recognizer.get_normalized_embedding(image, faces[0].landmarks)
```

---

## See Also

- [Detection Module](../modules/detection.md) - Built-in detection models
- [Recognition Module](../modules/recognition.md) - Built-in recognition models
- [Concepts: Overview](../concepts/overview.md) - Architecture overview
