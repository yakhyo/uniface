# Face Search

Find and identify people in images and video streams.

UniFace supports two search approaches:

| Approach             | Use case                                         | Tool                    |
| -------------------- | ------------------------------------------------ | ----------------------- |
| **Reference search** | "Is this specific person in the video?"          | `tools/search.py`       |
| **Vector search**    | "Who is this?" against a database of known faces | `tools/faiss_search.py` |

---

## Reference Search (single image)

Compare every detected face against a single reference photo:

```python
import cv2
import numpy as np
from uniface.detection import RetinaFace
from uniface.recognition import ArcFace
from uniface.face_utils import compute_similarity

detector = RetinaFace()
recognizer = ArcFace()

ref_image = cv2.imread("reference.jpg")
ref_faces = detector.detect(ref_image)
ref_embedding = recognizer.get_normalized_embedding(ref_image, ref_faces[0].landmarks)

query_image = cv2.imread("group_photo.jpg")
faces = detector.detect(query_image)

for face in faces:
    embedding = recognizer.get_normalized_embedding(query_image, face.landmarks)
    sim = compute_similarity(ref_embedding, embedding)

    label = f"Match ({sim:.2f})" if sim > 0.4 else f"Unknown ({sim:.2f})"
    print(label)
```

**CLI tool:**

```bash
python tools/search.py --reference ref.jpg --source video.mp4
python tools/search.py --reference ref.jpg --source 0  # webcam
```

---

## Vector Search (FAISS index)

For identifying faces against a database of many known people, use the
[`FAISS`](../modules/indexing.md) vector store.

!!! info "Install extra"
`bash
    pip install faiss-cpu
    `

### Build an index

Organise face images in person sub-folders:

```
dataset/
├── alice/
│   ├── 001.jpg
│   └── 002.jpg
├── bob/
│   └── 001.jpg
└── charlie/
    ├── 001.jpg
    └── 002.jpg
```

```python
import cv2
from pathlib import Path
from uniface.detection import RetinaFace
from uniface.recognition import ArcFace
from uniface.indexing import FAISS

detector = RetinaFace()
recognizer = ArcFace()
store = FAISS(db_path="./my_index")

for person_dir in sorted(Path("dataset").iterdir()):
    if not person_dir.is_dir():
        continue
    for img_path in person_dir.glob("*.jpg"):
        image = cv2.imread(str(img_path))
        faces = detector.detect(image)
        if faces:
            emb = recognizer.get_normalized_embedding(image, faces[0].landmarks)
            store.add(emb, {"person_id": person_dir.name, "source": str(img_path)})

store.save()
print(f"Index saved: {store}")
```

**CLI tool:**

```bash
python tools/faiss_search.py build --faces-dir dataset/ --db-path ./my_index
```

### Search against the index

```python
import cv2
from uniface.detection import RetinaFace
from uniface.recognition import ArcFace
from uniface.indexing import FAISS

detector = RetinaFace()
recognizer = ArcFace()

store = FAISS(db_path="./my_index")
store.load()

image = cv2.imread("query.jpg")
faces = detector.detect(image)

for face in faces:
    embedding = recognizer.get_normalized_embedding(image, face.landmarks)
    result, similarity = store.search(embedding, threshold=0.4)

    if result:
        print(f"Matched: {result['person_id']} ({similarity:.2f})")
    else:
        print(f"Unknown ({similarity:.2f})")
```

**CLI tool:**

```bash
python tools/faiss_search.py run --db-path ./my_index --source video.mp4
python tools/faiss_search.py run --db-path ./my_index --source 0  # webcam
```

### Manage the index

```python
from uniface.indexing import FAISS

store = FAISS(db_path="./my_index")
store.load()

print(f"Total vectors: {len(store)}")

removed = store.remove("person_id", "bob")
print(f"Removed {removed} entries")

store.save()
```

---

## See Also

- [Indexing Module](../modules/indexing.md) - Full `FAISS` API reference
- [Recognition Module](../modules/recognition.md) - Face recognition details
- [Video & Webcam](video-webcam.md) - Real-time processing
- [Concepts: Thresholds](../concepts/thresholds-calibration.md) - Tuning similarity thresholds
