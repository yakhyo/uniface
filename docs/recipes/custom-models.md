# Custom Models

Add your own ONNX models to UniFace.

---

## Add Detection Model

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
        # Preprocess
        input_tensor = self._preprocess(image)

        # Inference
        outputs = self.session.run(None, {'input': input_tensor})

        # Postprocess
        faces = self._postprocess(outputs, image.shape)
        return faces

    def _preprocess(self, image):
        # Your preprocessing logic
        pass

    def _postprocess(self, outputs, shape):
        # Your postprocessing logic
        pass
```

---

## Add Recognition Model

```python
from uniface.recognition.base import BaseRecognizer
from uniface.onnx_utils import create_onnx_session
from uniface import face_alignment
import numpy as np

class MyRecognizer(BaseRecognizer):
    def __init__(self, model_path: str):
        self.session = create_onnx_session(model_path)

    def get_normalized_embedding(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        # Align face
        aligned = face_alignment(image, landmarks)

        # Preprocess
        input_tensor = self._preprocess(aligned)

        # Inference
        embedding = self.session.run(None, {'input': input_tensor})[0]

        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def _preprocess(self, image):
        # Your preprocessing logic
        pass
```

---

## Register Weights

Add to `uniface/constants.py`:

```python
class MyModelWeights(str, Enum):
    DEFAULT = "my_model"

MODEL_URLS[MyModelWeights.DEFAULT] = 'https://...'
MODEL_SHA256[MyModelWeights.DEFAULT] = 'sha256hash...'
```

---

## Use Custom Model

```python
from my_module import MyDetector

detector = MyDetector("path/to/model.onnx")
faces = detector.detect(image)
```
