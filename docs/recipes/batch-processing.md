# Batch Processing

Process multiple images efficiently.

!!! note "Work in Progress"
    This page contains example code patterns. Test thoroughly before using in production.

---

## Basic Batch Processing

```python
import cv2
from pathlib import Path
from uniface import RetinaFace

detector = RetinaFace()

def process_directory(input_dir, output_dir):
    """Process all images in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for image_path in input_path.glob("*.jpg"):
        print(f"Processing {image_path.name}...")

        image = cv2.imread(str(image_path))
        faces = detector.detect(image)

        print(f"  Found {len(faces)} face(s)")

        # Process and save results
        # ... your code here ...

# Usage
process_directory("input_images/", "output_images/")
```

---

## With Progress Bar

```python
from tqdm import tqdm

for image_path in tqdm(image_files, desc="Processing"):
    # ... process image ...
    pass
```

---

## Extract Embeddings

```python
from uniface import RetinaFace, ArcFace
import numpy as np

detector = RetinaFace()
recognizer = ArcFace()

embeddings = {}
for image_path in Path("faces/").glob("*.jpg"):
    image = cv2.imread(str(image_path))
    faces = detector.detect(image)

    if faces:
        embedding = recognizer.get_normalized_embedding(image, faces[0].landmarks)
        embeddings[image_path.stem] = embedding

# Save embeddings
np.savez("embeddings.npz", **embeddings)
```

---

## See Also

- [Video & Webcam](video-webcam.md) - Real-time processing
- [Face Search](face-search.md) - Search through embeddings
- [Image Pipeline](image-pipeline.md) - Full analysis pipeline
- [Detection Module](../modules/detection.md) - Detection options
