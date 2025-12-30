# Image Pipeline

A complete pipeline for processing images with detection, recognition, and attribute analysis.

---

## Basic Pipeline

```python
import cv2
from uniface import RetinaFace, ArcFace, AgeGender
from uniface.visualization import draw_detections

# Initialize models
detector = RetinaFace()
recognizer = ArcFace()
age_gender = AgeGender()

def process_image(image_path):
    """Process a single image through the full pipeline."""
    # Load image
    image = cv2.imread(image_path)

    # Step 1: Detect faces
    faces = detector.detect(image)
    print(f"Found {len(faces)} face(s)")

    results = []

    for i, face in enumerate(faces):
        # Step 2: Extract embedding
        embedding = recognizer.get_normalized_embedding(image, face.landmarks)

        # Step 3: Predict attributes
        attrs = age_gender.predict(image, face.bbox)

        results.append({
            'face_id': i,
            'bbox': face.bbox,
            'confidence': face.confidence,
            'embedding': embedding,
            'gender': attrs.sex,
            'age': attrs.age
        })

        print(f"  Face {i+1}: {attrs.sex}, {attrs.age} years old")

    # Visualize
    draw_detections(
        image=image,
        bboxes=[f.bbox for f in faces],
        scores=[f.confidence for f in faces],
        landmarks=[f.landmarks for f in faces]
    )

    return image, results

# Usage
result_image, results = process_image("photo.jpg")
cv2.imwrite("result.jpg", result_image)
```

---

## Using FaceAnalyzer

For convenience, use the built-in `FaceAnalyzer`:

```python
from uniface import FaceAnalyzer
import cv2

# Initialize with desired modules
analyzer = FaceAnalyzer(
    detect=True,
    recognize=True,
    attributes=True
)

# Process image
image = cv2.imread("photo.jpg")
faces = analyzer.analyze(image)

# Access enriched Face objects
for face in faces:
    print(f"Confidence: {face.confidence:.2f}")
    print(f"Embedding: {face.embedding.shape}")
    print(f"Age: {face.age}, Gender: {face.sex}")
```

---

## Full Analysis Pipeline

Complete pipeline with all modules:

```python
import cv2
import numpy as np
from uniface import (
    RetinaFace, ArcFace, AgeGender, FairFace,
    Landmark106, MobileGaze
)
from uniface.parsing import BiSeNet
from uniface.spoofing import MiniFASNet
from uniface.visualization import draw_detections, draw_gaze

class FaceAnalysisPipeline:
    def __init__(self):
        # Initialize all models
        self.detector = RetinaFace()
        self.recognizer = ArcFace()
        self.age_gender = AgeGender()
        self.fairface = FairFace()
        self.landmarker = Landmark106()
        self.gaze = MobileGaze()
        self.parser = BiSeNet()
        self.spoofer = MiniFASNet()

    def analyze(self, image):
        """Run full analysis pipeline."""
        faces = self.detector.detect(image)
        results = []

        for face in faces:
            result = {
                'bbox': face.bbox,
                'confidence': face.confidence,
                'landmarks_5': face.landmarks
            }

            # Recognition embedding
            result['embedding'] = self.recognizer.get_normalized_embedding(
                image, face.landmarks
            )

            # Attributes
            ag_result = self.age_gender.predict(image, face.bbox)
            result['age'] = ag_result.age
            result['gender'] = ag_result.sex

            # FairFace attributes
            ff_result = self.fairface.predict(image, face.bbox)
            result['age_group'] = ff_result.age_group
            result['race'] = ff_result.race

            # 106-point landmarks
            result['landmarks_106'] = self.landmarker.get_landmarks(
                image, face.bbox
            )

            # Gaze estimation
            x1, y1, x2, y2 = map(int, face.bbox)
            face_crop = image[y1:y2, x1:x2]
            if face_crop.size > 0:
                gaze_result = self.gaze.estimate(face_crop)
                result['gaze_pitch'] = gaze_result.pitch
                result['gaze_yaw'] = gaze_result.yaw

            # Face parsing
            if face_crop.size > 0:
                result['parsing_mask'] = self.parser.parse(face_crop)

            # Anti-spoofing
            spoof_result = self.spoofer.predict(image, face.bbox)
            result['is_real'] = spoof_result.is_real
            result['spoof_confidence'] = spoof_result.confidence

            results.append(result)

        return results

# Usage
pipeline = FaceAnalysisPipeline()
results = pipeline.analyze(cv2.imread("photo.jpg"))

for i, r in enumerate(results):
    print(f"\nFace {i+1}:")
    print(f"  Gender: {r['gender']}, Age: {r['age']}")
    print(f"  Race: {r['race']}, Age Group: {r['age_group']}")
    print(f"  Gaze: pitch={np.degrees(r['gaze_pitch']):.1f}°")
    print(f"  Real: {r['is_real']} ({r['spoof_confidence']:.1%})")
```

---

## Visualization Pipeline

```python
import cv2
import numpy as np
from uniface import RetinaFace, AgeGender, MobileGaze
from uniface.visualization import draw_detections, draw_gaze

def visualize_analysis(image_path, output_path):
    """Create annotated visualization of face analysis."""
    detector = RetinaFace()
    age_gender = AgeGender()
    gaze = MobileGaze()

    image = cv2.imread(image_path)
    faces = detector.detect(image)

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Age and gender
        attrs = age_gender.predict(image, face.bbox)
        label = f"{attrs.sex}, {attrs.age}y"
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Gaze
        face_crop = image[y1:y2, x1:x2]
        if face_crop.size > 0:
            gaze_result = gaze.estimate(face_crop)
            draw_gaze(image, face.bbox, gaze_result.pitch, gaze_result.yaw)

        # Confidence
        conf_label = f"{face.confidence:.0%}"
        cv2.putText(image, conf_label, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(output_path, image)
    print(f"Saved to {output_path}")

# Usage
visualize_analysis("input.jpg", "output.jpg")
```

---

## JSON Output

Export results to JSON:

```python
import json
import numpy as np

def results_to_json(results):
    """Convert analysis results to JSON-serializable format."""
    output = []

    for r in results:
        item = {
            'bbox': r['bbox'].tolist(),
            'confidence': float(r['confidence']),
            'age': int(r['age']) if r.get('age') else None,
            'gender': r.get('gender'),
            'race': r.get('race'),
            'is_real': r.get('is_real'),
            'gaze': {
                'pitch_deg': float(np.degrees(r['gaze_pitch'])) if 'gaze_pitch' in r else None,
                'yaw_deg': float(np.degrees(r['gaze_yaw'])) if 'gaze_yaw' in r else None
            }
        }
        output.append(item)

    return output

# Usage
results = pipeline.analyze(image)
json_data = results_to_json(results)

with open('results.json', 'w') as f:
    json.dump(json_data, f, indent=2)
```

---

## Next Steps

- [Batch Processing](batch-processing.md) - Process multiple images
- [Video & Webcam](video-webcam.md) - Real-time processing
- [Face Search](face-search.md) - Build a search system
