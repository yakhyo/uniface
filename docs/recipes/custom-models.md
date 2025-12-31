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
        self.session = create_onnx_session(model_path)
        self.threshold = confidence_threshold

    def detect(self, image: np.ndarray) -> list[Face]:
        # 1. Preprocess image
        input_tensor = self._preprocess(image)

        # 2. Run inference
        outputs = self.session.run(None, {'input': input_tensor})

        # 3. Postprocess outputs to Face objects
        faces = self._postprocess(outputs, image.shape)
        return faces

    def _preprocess(self, image):
        # Your preprocessing logic
        # e.g., resize, normalize, transpose
        pass

    def _postprocess(self, outputs, shape):
        # Your postprocessing logic
        # e.g., decode boxes, apply NMS, create Face objects
        pass
```

---

## Add Custom Recognition Model

```python
from uniface.recognition.base import BaseRecognizer
from uniface.onnx_utils import create_onnx_session
from uniface import face_alignment
import numpy as np

class MyRecognizer(BaseRecognizer):
    def __init__(self, model_path: str):
        self.session = create_onnx_session(model_path)

    def get_normalized_embedding(
        self,
        image: np.ndarray,
        landmarks: np.ndarray
    ) -> np.ndarray:
        # 1. Align face
        aligned = face_alignment(image, landmarks)

        # 2. Preprocess
        input_tensor = self._preprocess(aligned)

        # 3. Run inference
        embedding = self.session.run(None, {'input': input_tensor})[0]

        # 4. Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def _preprocess(self, image):
        # Your preprocessing logic
        pass
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
