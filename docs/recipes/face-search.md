# Face Search

Build a face search system for finding people in images.

---

## Build Face Database

```python
import numpy as np
import cv2
from pathlib import Path
from uniface import RetinaFace, ArcFace

class FaceDatabase:
    def __init__(self):
        self.detector = RetinaFace()
        self.recognizer = ArcFace()
        self.embeddings = {}
        self.metadata = {}

    def add_face(self, person_id, image, metadata=None):
        """Add a face to the database."""
        faces = self.detector.detect(image)

        if not faces:
            raise ValueError(f"No face found for {person_id}")

        # Use highest confidence face
        face = max(faces, key=lambda f: f.confidence)
        embedding = self.recognizer.get_normalized_embedding(image, face.landmarks)

        self.embeddings[person_id] = embedding
        self.metadata[person_id] = metadata or {}

        return True

    def add_from_directory(self, directory):
        """Add faces from a directory (filename = person_id)."""
        dir_path = Path(directory)

        for image_path in dir_path.glob("*.jpg"):
            person_id = image_path.stem
            image = cv2.imread(str(image_path))

            try:
                self.add_face(person_id, image, {'source': str(image_path)})
                print(f"Added: {person_id}")
            except ValueError as e:
                print(f"Skipped {person_id}: {e}")

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
                'confidence': face.confidence,
                'match': best_match if best_similarity >= threshold else None,
                'similarity': best_similarity,
                'metadata': self.metadata.get(best_match, {})
            })

        return results

    def save(self, path):
        """Save database to file."""
        np.savez(
            path,
            embeddings=dict(self.embeddings),
            metadata=self.metadata
        )
        print(f"Saved database to {path}")

    def load(self, path):
        """Load database from file."""
        data = np.load(path, allow_pickle=True)
        self.embeddings = data['embeddings'].item()
        self.metadata = data['metadata'].item()
        print(f"Loaded {len(self.embeddings)} faces from {path}")

# Usage
db = FaceDatabase()

# Add faces from directory
db.add_from_directory("known_faces/")

# Save for later
db.save("face_database.npz")

# Search for person
query_image = cv2.imread("group_photo.jpg")
results = db.search(query_image)

for r in results:
    if r['match']:
        print(f"Found: {r['match']} (similarity: {r['similarity']:.3f})")
    else:
        print(f"Unknown face (best similarity: {r['similarity']:.3f})")
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

## Top-K Search

Find top K matches instead of best match only:

```python
def search_top_k(self, embedding, k=5):
    """Find top K matches for an embedding."""
    similarities = []

    for person_id, db_embedding in self.embeddings.items():
        similarity = np.dot(embedding, db_embedding.T)[0][0]
        similarities.append((person_id, similarity))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:k]

# Usage
query_embedding = recognizer.get_normalized_embedding(image, face.landmarks)
top_matches = search_top_k(query_embedding, k=3)

for person_id, similarity in top_matches:
    print(f"{person_id}: {similarity:.4f}")
```

---

## Batch Search

Search through multiple query images:

```python
from pathlib import Path

def batch_search(db, query_dir, threshold=0.6):
    """Search for faces in multiple images."""
    all_results = {}

    for image_path in Path(query_dir).glob("*.jpg"):
        image = cv2.imread(str(image_path))
        results = db.search(image, threshold)

        matches = [r['match'] for r in results if r['match']]
        all_results[image_path.name] = matches

        print(f"{image_path.name}: {matches}")

    return all_results

# Usage
results = batch_search(db, "query_images/")
```

---

## Find Person in Group Photo

```python
def find_person_in_group(db, person_id, group_image, threshold=0.6):
    """Find a specific person in a group photo."""
    if person_id not in db.embeddings:
        raise ValueError(f"Person {person_id} not in database")

    reference_embedding = db.embeddings[person_id]
    faces = db.detector.detect(group_image)

    best_match = None
    best_similarity = -1

    for face in faces:
        embedding = db.recognizer.get_normalized_embedding(
            group_image, face.landmarks
        )
        similarity = np.dot(embedding, reference_embedding.T)[0][0]

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = face

    if best_match and best_similarity >= threshold:
        return {
            'found': True,
            'face': best_match,
            'similarity': best_similarity
        }

    return {'found': False, 'similarity': best_similarity}

# Usage
group = cv2.imread("group_photo.jpg")
result = find_person_in_group(db, "john_doe", group)

if result['found']:
    print(f"Found with similarity: {result['similarity']:.3f}")
    # Draw the found face
    x1, y1, x2, y2 = map(int, result['face'].bbox)
    cv2.rectangle(group, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.imwrite("found.jpg", group)
```

---

## Update Database

Add or update faces:

```python
def update_face(db, person_id, new_image):
    """Update a person's face in the database."""
    faces = db.detector.detect(new_image)

    if not faces:
        print(f"No face found in new image for {person_id}")
        return False

    face = max(faces, key=lambda f: f.confidence)
    new_embedding = db.recognizer.get_normalized_embedding(
        new_image, face.landmarks
    )

    if person_id in db.embeddings:
        # Average with existing embedding
        old_embedding = db.embeddings[person_id]
        db.embeddings[person_id] = (old_embedding + new_embedding) / 2
        # Re-normalize
        db.embeddings[person_id] /= np.linalg.norm(db.embeddings[person_id])
        print(f"Updated: {person_id}")
    else:
        db.embeddings[person_id] = new_embedding
        print(f"Added: {person_id}")

    return True

# Usage
update_face(db, "john_doe", cv2.imread("john_new.jpg"))
db.save("face_database.npz")
```

---

## Next Steps

- [Anonymize Stream](anonymize-stream.md) - Privacy protection
- [Batch Processing](batch-processing.md) - Process multiple files
- [Recognition Module](../modules/recognition.md) - Model details
