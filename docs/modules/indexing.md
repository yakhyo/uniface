# Indexing

FAISS-backed vector store for fast similarity search over embeddings.

!!! info "Optional dependency"
    ```bash
    pip install faiss-cpu
    ```

---

## FAISS

```python
from uniface.indexing import FAISS
```

A thin wrapper around a FAISS `IndexFlatIP` (inner-product) index. Vectors
**must** be L2-normalised before adding so that inner product equals cosine
similarity. The store does not normalise internally.

Each vector is paired with a metadata `dict` that can carry any
JSON-serialisable payload (person ID, name, source path, etc.).

### Constructor

```python
store = FAISS(embedding_size=512, db_path="./vector_index")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_size` | `int` | `512` | Dimension of embedding vectors |
| `db_path` | `str` | `"./vector_index"` | Directory for persisting index and metadata |

---

### Methods

#### `add(embedding, metadata)`

Add a single embedding with associated metadata.

```python
store.add(embedding, {"person_id": "alice", "source": "photo.jpg"})
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `embedding` | `np.ndarray` | L2-normalised embedding vector |
| `metadata` | `dict[str, Any]` | Arbitrary JSON-serialisable key-value pairs |

---

#### `search(embedding, threshold=0.4)`

Find the closest match for a query embedding.

```python
result, similarity = store.search(query_embedding, threshold=0.4)
if result:
    print(result["person_id"], similarity)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding` | `np.ndarray` | — | L2-normalised query vector |
| `threshold` | `float` | `0.4` | Minimum cosine similarity to accept a match |

**Returns:** `(metadata, similarity)` if a match is found, or `(None, similarity)` when below threshold or the index is empty.

---

#### `remove(key, value)`

Remove all entries where `metadata[key] == value` and rebuild the index.

```python
removed = store.remove("person_id", "bob")
print(f"Removed {removed} entries")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | `str` | Metadata key to match |
| `value` | `Any` | Value to match |

**Returns:** Number of entries removed.

---

#### `save()`

Persist the FAISS index and metadata to disk.

```python
store.save()
```

Writes two files to `db_path`:

- `faiss_index.bin` — binary FAISS index
- `metadata.json` — JSON array of metadata dicts

---

#### `load()`

Load a previously saved index and metadata.

```python
store = FAISS(db_path="./vector_index")
loaded = store.load()  # True if files exist
```

**Returns:** `True` if loaded successfully, `False` if files are missing.

**Raises:** `RuntimeError` if files exist but cannot be read.

---

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `size` | `int` | Number of vectors in the index |
| `len(store)` | `int` | Same as `size` |

---

## Example: End-to-End

```python
import cv2
from uniface.detection import RetinaFace
from uniface.recognition import ArcFace
from uniface.indexing import FAISS

detector = RetinaFace()
recognizer = ArcFace()

# Build
store = FAISS(db_path="./my_index")

image = cv2.imread("alice.jpg")
faces = detector.detect(image)
embedding = recognizer.get_normalized_embedding(image, faces[0].landmarks)
store.add(embedding, {"person_id": "alice"})
store.save()

# Search
store2 = FAISS(db_path="./my_index")
store2.load()

query = cv2.imread("unknown.jpg")
faces = detector.detect(query)
emb = recognizer.get_normalized_embedding(query, faces[0].landmarks)

result, sim = store2.search(emb)
if result:
    print(f"Matched: {result['person_id']} (similarity: {sim:.3f})")
else:
    print(f"No match (similarity: {sim:.3f})")
```

---

## See Also

- [Face Search Recipe](../recipes/face-search.md) - Building and querying indexes
- [Recognition Module](recognition.md) - Embedding extraction
- [Thresholds Guide](../concepts/thresholds-calibration.md) - Tuning similarity thresholds
