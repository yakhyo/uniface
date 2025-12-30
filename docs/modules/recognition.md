# Recognition

Face recognition extracts embeddings for identity verification and face search.

---

## Available Models

| Model | Backbone | Size | Embedding Dim | Best For |
|-------|----------|------|---------------|----------|
| **ArcFace** | MobileNet/ResNet | 8-166 MB | 512 | General use (recommended) |
| **MobileFace** | MobileNet V2/V3 | 1-10 MB | 512 | Mobile/Edge |
| **SphereFace** | Sphere20/36 | 50-92 MB | 512 | Research |

---

## ArcFace

State-of-the-art recognition using additive angular margin loss.

### Basic Usage

```python
from uniface import RetinaFace, ArcFace

detector = RetinaFace()
recognizer = ArcFace()

# Detect face
faces = detector.detect(image)

# Extract embedding
if faces:
    embedding = recognizer.get_normalized_embedding(image, faces[0].landmarks)
    print(f"Embedding shape: {embedding.shape}")  # (1, 512)
```

### Model Variants

```python
from uniface import ArcFace
from uniface.constants import ArcFaceWeights

# Lightweight (default)
recognizer = ArcFace(model_name=ArcFaceWeights.MNET)

# High accuracy
recognizer = ArcFace(model_name=ArcFaceWeights.RESNET)
```

| Variant | Backbone | Size | Use Case |
|---------|----------|------|----------|
| **MNET** ⭐ | MobileNet | 8 MB | Balanced (recommended) |
| RESNET | ResNet50 | 166 MB | Maximum accuracy |

---

## MobileFace

Lightweight recognition for resource-constrained environments.

### Basic Usage

```python
from uniface import MobileFace

recognizer = MobileFace()
embedding = recognizer.get_normalized_embedding(image, landmarks)
```

### Model Variants

```python
from uniface import MobileFace
from uniface.constants import MobileFaceWeights

# Ultra-lightweight
recognizer = MobileFace(model_name=MobileFaceWeights.MNET_025)

# Balanced (default)
recognizer = MobileFace(model_name=MobileFaceWeights.MNET_V2)

# Higher accuracy
recognizer = MobileFace(model_name=MobileFaceWeights.MNET_V3_LARGE)
```

| Variant | Params | Size | LFW | Use Case |
|---------|--------|------|-----|----------|
| MNET_025 | 0.36M | 1 MB | 98.8% | Ultra-lightweight |
| **MNET_V2** ⭐ | 2.29M | 4 MB | 99.6% | Mobile/Edge |
| MNET_V3_SMALL | 1.25M | 3 MB | 99.3% | Mobile optimized |
| MNET_V3_LARGE | 3.52M | 10 MB | 99.5% | Balanced mobile |

---

## SphereFace

Recognition using angular softmax loss (A-Softmax).

### Basic Usage

```python
from uniface import SphereFace
from uniface.constants import SphereFaceWeights

recognizer = SphereFace(model_name=SphereFaceWeights.SPHERE20)
embedding = recognizer.get_normalized_embedding(image, landmarks)
```

| Variant | Params | Size | LFW | Use Case |
|---------|--------|------|-----|----------|
| SPHERE20 | 24.5M | 50 MB | 99.7% | Research |
| SPHERE36 | 34.6M | 92 MB | 99.7% | Research |

---

## Face Comparison

### Compute Similarity

```python
from uniface import compute_similarity
import numpy as np

# Extract embeddings
emb1 = recognizer.get_normalized_embedding(image1, landmarks1)
emb2 = recognizer.get_normalized_embedding(image2, landmarks2)

# Method 1: Using utility function
similarity = compute_similarity(emb1, emb2)

# Method 2: Direct computation
similarity = np.dot(emb1, emb2.T)[0][0]

print(f"Similarity: {similarity:.4f}")
```

### Threshold Guidelines

| Threshold | Decision | Use Case |
|-----------|----------|----------|
| > 0.7 | Very high confidence | Security-critical |
| > 0.6 | Same person | General verification |
| 0.4 - 0.6 | Uncertain | Manual review needed |
| < 0.4 | Different people | Rejection |

---

## Face Alignment

Recognition models require aligned faces. UniFace handles this internally:

```python
# Alignment is done automatically
embedding = recognizer.get_normalized_embedding(image, landmarks)

# Or manually align
from uniface import face_alignment

aligned_face = face_alignment(image, landmarks)
# Returns: 112x112 aligned face image
```

---

## Building a Face Database

```python
import numpy as np
from uniface import RetinaFace, ArcFace

detector = RetinaFace()
recognizer = ArcFace()

# Build database
database = {}
for person_id, image_path in person_images.items():
    image = cv2.imread(image_path)
    faces = detector.detect(image)

    if faces:
        embedding = recognizer.get_normalized_embedding(image, faces[0].landmarks)
        database[person_id] = embedding

# Save for later use
np.savez('face_database.npz', **database)

# Load database
data = np.load('face_database.npz')
database = {key: data[key] for key in data.files}
```

---

## Face Search

Find a person in a database:

```python
def search_face(query_embedding, database, threshold=0.6):
    """Find best match in database."""
    best_match = None
    best_similarity = -1

    for person_id, db_embedding in database.items():
        similarity = np.dot(query_embedding, db_embedding.T)[0][0]

        if similarity > best_similarity and similarity > threshold:
            best_similarity = similarity
            best_match = person_id

    return best_match, best_similarity

# Usage
query_embedding = recognizer.get_normalized_embedding(query_image, landmarks)
match, similarity = search_face(query_embedding, database)

if match:
    print(f"Found: {match} (similarity: {similarity:.4f})")
else:
    print("No match found")
```

---

## Factory Function

```python
from uniface import create_recognizer

recognizer = create_recognizer('arcface')
```

---

## Next Steps

- [Landmarks](landmarks.md) - 106-point landmarks
- [Face Search Recipe](../recipes/face-search.md) - Complete search system
- [Thresholds](../concepts/thresholds-calibration.md) - Calibration guide
