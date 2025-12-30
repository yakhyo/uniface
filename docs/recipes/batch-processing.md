# Batch Processing

Process multiple images efficiently.

---

## Basic Batch Processing

```python
import cv2
from pathlib import Path
from uniface import RetinaFace
from uniface.visualization import draw_detections

detector = RetinaFace()

def process_directory(input_dir, output_dir):
    """Process all images in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Supported image formats
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(ext))
        image_files.extend(input_path.glob(ext.upper()))

    print(f"Found {len(image_files)} images")

    results = {}

    for image_path in image_files:
        print(f"Processing {image_path.name}...")

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  Failed to load {image_path.name}")
            continue

        faces = detector.detect(image)
        print(f"  Found {len(faces)} face(s)")

        # Store results
        results[image_path.name] = {
            'num_faces': len(faces),
            'faces': [
                {
                    'bbox': face.bbox.tolist(),
                    'confidence': float(face.confidence)
                }
                for face in faces
            ]
        }

        # Visualize and save
        if faces:
            draw_detections(
                image=image,
                bboxes=[f.bbox for f in faces],
                scores=[f.confidence for f in faces],
                landmarks=[f.landmarks for f in faces]
            )

        output_file = output_path / image_path.name
        cv2.imwrite(str(output_file), image)

    return results

# Usage
results = process_directory("input_images/", "output_images/")
print(f"\nProcessed {len(results)} images")
```

---

## Parallel Processing

Use multiprocessing for faster batch processing:

```python
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from uniface import RetinaFace

def process_single_image(image_path, output_dir):
    """Process a single image (runs in worker process)."""
    # Create detector in each process
    detector = RetinaFace()

    image = cv2.imread(str(image_path))
    if image is None:
        return image_path.name, {'error': 'Failed to load'}

    faces = detector.detect(image)

    result = {
        'num_faces': len(faces),
        'faces': [
            {
                'bbox': face.bbox.tolist(),
                'confidence': float(face.confidence)
            }
            for face in faces
        ]
    }

    # Save result
    output_path = Path(output_dir) / image_path.name
    cv2.imwrite(str(output_path), image)

    return image_path.name, result

def batch_process_parallel(input_dir, output_dir, max_workers=4):
    """Process images in parallel."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))

    results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_image, img, output_dir): img
            for img in image_files
        }

        for future in as_completed(futures):
            name, result = future.result()
            results[name] = result
            print(f"Completed: {name} - {result.get('num_faces', 'error')} faces")

    return results

# Usage
results = batch_process_parallel("input_images/", "output_images/", max_workers=4)
```

---

## Progress Tracking

Use tqdm for progress bars:

```python
from tqdm import tqdm

def process_with_progress(input_dir, output_dir):
    """Process with progress bar."""
    detector = RetinaFace()

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))

    results = {}

    for image_path in tqdm(image_files, desc="Processing images"):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        faces = detector.detect(image)
        results[image_path.name] = len(faces)

        cv2.imwrite(str(output_path / image_path.name), image)

    return results

# Usage
results = process_with_progress("input/", "output/")
print(f"Total faces found: {sum(results.values())}")
```

---

## Batch Embedding Extraction

Extract embeddings for a face database:

```python
import numpy as np
from pathlib import Path
from uniface import RetinaFace, ArcFace

def extract_embeddings(image_dir):
    """Extract embeddings from all faces."""
    detector = RetinaFace()
    recognizer = ArcFace()

    embeddings = {}

    for image_path in Path(image_dir).glob("*.jpg"):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        faces = detector.detect(image)

        if faces:
            # Use first face
            embedding = recognizer.get_normalized_embedding(
                image, faces[0].landmarks
            )
            embeddings[image_path.stem] = embedding
            print(f"Extracted: {image_path.stem}")

    return embeddings

def save_embeddings(embeddings, output_path):
    """Save embeddings to file."""
    np.savez(output_path, **embeddings)
    print(f"Saved {len(embeddings)} embeddings to {output_path}")

def load_embeddings(input_path):
    """Load embeddings from file."""
    data = np.load(input_path)
    return {key: data[key] for key in data.files}

# Usage
embeddings = extract_embeddings("faces/")
save_embeddings(embeddings, "embeddings.npz")

# Later...
loaded = load_embeddings("embeddings.npz")
```

---

## CSV Output

Export results to CSV:

```python
import csv
from pathlib import Path

def export_to_csv(results, output_path):
    """Export detection results to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'face_id', 'x1', 'y1', 'x2', 'y2', 'confidence'])

        for filename, data in results.items():
            for i, face in enumerate(data['faces']):
                bbox = face['bbox']
                writer.writerow([
                    filename, i,
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    face['confidence']
                ])

    print(f"Exported to {output_path}")

# Usage
results = process_directory("input/", "output/")
export_to_csv(results, "detections.csv")
```

---

## Memory-Efficient Processing

For large batches, process in chunks:

```python
def process_in_chunks(image_files, chunk_size=100):
    """Process images in memory-efficient chunks."""
    detector = RetinaFace()

    all_results = {}

    for i in range(0, len(image_files), chunk_size):
        chunk = image_files[i:i + chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}/{(len(image_files)-1)//chunk_size + 1}")

        for image_path in chunk:
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            faces = detector.detect(image)
            all_results[image_path.name] = len(faces)

            # Free memory
            del image

        # Optional: force garbage collection
        import gc
        gc.collect()

    return all_results
```

---

## Error Handling

Robust batch processing with error handling:

```python
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def robust_batch_process(input_dir, output_dir):
    """Batch process with error handling."""
    detector = RetinaFace()

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = list(input_path.glob("*.[jJ][pP][gG]"))

    success_count = 0
    error_count = 0

    for image_path in image_files:
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("Failed to load image")

            faces = detector.detect(image)

            cv2.imwrite(str(output_path / image_path.name), image)
            success_count += 1
            logger.info(f"Processed {image_path.name}: {len(faces)} faces")

        except Exception as e:
            error_count += 1
            logger.error(f"Error processing {image_path.name}: {e}")

    logger.info(f"Completed: {success_count} success, {error_count} errors")
    return success_count, error_count
```

---

## Next Steps

- [Video & Webcam](video-webcam.md) - Real-time processing
- [Face Search](face-search.md) - Search through embeddings
- [Image Pipeline](image-pipeline.md) - Full analysis pipeline
