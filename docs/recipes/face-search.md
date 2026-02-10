# Face Search

Build a face search system for finding people in images.

!!! note "Work in Progress"
    This page contains example code patterns. Test thoroughly before using in production.

---

## Basic Face Database

```python
import numpy as np
import cv2
from pathlib import Path
from uniface.detection import RetinaFace
from uniface.recognition import ArcFace

class FaceDatabase:
    def __init__(self):
        self.detector = RetinaFace()
        self.recognizer = ArcFace()
        self.embeddings = {}

    def add_face(self, person_id, image):
        """Add a face to the database."""
        faces = self.detector.detect(image)
        if not faces:
            raise ValueError(f"No face found for {person_id}")

        face = max(faces, key=lambda f: f.confidence)
        embedding = self.recognizer.get_normalized_embedding(image, face.landmarks)
        self.embeddings[person_id] = embedding
        return True

    def search(self, image, threshold=0.6):
        """Search for faces in an image."""
        faces = self.detector.detect(image)
        results = []

        for face in faces:
            embedding = self.recognizer.get_normalized_embedding(image, face.landmarks)

            best_match = None
            best_similarity = -1

            for person_id, db_embedding in self.embeddings.items():
                similarity = np.dot(embedding, db_embedding.T)[0][0]
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = person_id

            results.append({
                'bbox': face.bbox,
                'match': best_match if best_similarity >= threshold else None,
                'similarity': best_similarity
            })

        return results

    def save(self, path):
        """Save database to file."""
        np.savez(path, embeddings=dict(self.embeddings))

    def load(self, path):
        """Load database from file."""
        data = np.load(path, allow_pickle=True)
        self.embeddings = data['embeddings'].item()

# Usage
db = FaceDatabase()

# Add faces
for image_path in Path("known_faces/").glob("*.jpg"):
    person_id = image_path.stem
    image = cv2.imread(str(image_path))
    try:
        db.add_face(person_id, image)
        print(f"Added: {person_id}")
    except ValueError as e:
        print(f"Skipped: {e}")

# Save database
db.save("face_database.npz")

# Search
query_image = cv2.imread("group_photo.jpg")
results = db.search(query_image)

for r in results:
    if r['match']:
        print(f"Found: {r['match']} (similarity: {r['similarity']:.3f})")
```

---

## Visualization

```python
import cv2

def visualize_search_results(image, results):
    """Draw search results on image."""
    for r in results:
        x1, y1, x2, y2 = map(int, r['bbox'])

        if r['match']:
            color = (0, 255, 0)  # Green for match
            label = f"{r['match']} ({r['similarity']:.2f})"
        else:
            color = (0, 0, 255)  # Red for unknown
            label = f"Unknown ({r['similarity']:.2f})"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image

# Usage
results = db.search(image)
annotated = visualize_search_results(image.copy(), results)
cv2.imwrite("search_result.jpg", annotated)
```

---

## Real-Time Search

```python
import cv2

def realtime_search(db):
    """Real-time face search from webcam."""
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = db.search(frame, threshold=0.5)

        for r in results:
            x1, y1, x2, y2 = map(int, r['bbox'])

            if r['match']:
                color = (0, 255, 0)
                label = r['match']
            else:
                color = (0, 0, 255)
                label = "Unknown"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Face Search", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Usage
db = FaceDatabase()
db.load("face_database.npz")
realtime_search(db)
```

---

## See Also

- [Recognition Module](../modules/recognition.md) - Face recognition details
- [Batch Processing](batch-processing.md) - Process multiple files
- [Video & Webcam](video-webcam.md) - Real-time processing
- [Concepts: Thresholds](../concepts/thresholds-calibration.md) - Tuning similarity thresholds
