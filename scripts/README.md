# Scripts

Collection of example scripts demonstrating UniFace functionality.

## Available Scripts

- `run_detection.py` - Face detection on images
- `run_age_gender.py` - Age and gender prediction
- `run_landmarks.py` - Facial landmark detection
- `run_recognition.py` - Face recognition and embeddings
- `run_face_search.py` - Face search and matching
- `run_video_detection.py` - Video processing with face detection
- `batch_process.py` - Batch processing of image folders
- `download_model.py` - Download and manage models

## Quick Start

```bash
# Face detection
python scripts/run_detection.py --image assets/test.jpg

# Age and gender detection
python scripts/run_age_gender.py --image assets/test.jpg

# Webcam demo
python scripts/run_age_gender.py --webcam

# Batch processing
python scripts/batch_process.py --input images/ --output results/
```

## Import Examples

The scripts use direct class imports for better developer experience:

```python
# Face Detection
from uniface.detection import RetinaFace, SCRFD

detector = RetinaFace()  # or SCRFD()
faces = detector.detect(image)

# Face Recognition
from uniface.recognition import ArcFace, MobileFace, SphereFace

recognizer = ArcFace()  # or MobileFace(), SphereFace()
embedding = recognizer.get_embedding(image, landmarks)

# Age & Gender
from uniface.attribute import AgeGender

age_gender = AgeGender()
gender, age = age_gender.predict(image, bbox)

# Landmarks
from uniface.landmark import Landmark106

landmarker = Landmark106()
landmarks = landmarker.get_landmarks(image, bbox)
```

## Available Classes

**Detection:**
- `RetinaFace` - High accuracy face detection
- `SCRFD` - Fast face detection

**Recognition:**
- `ArcFace` - High accuracy face recognition
- `MobileFace` - Lightweight face recognition
- `SphereFace` - Alternative face recognition

**Attributes:**
- `AgeGender` - Age and gender prediction

**Landmarks:**
- `Landmark106` - 106-point facial landmarks

## Common Options

Most scripts support:
- `--help` - Show usage information
- `--verbose` - Enable detailed logging
- `--detector` - Choose detector (retinaface, scrfd)
- `--threshold` - Set confidence threshold

## Testing

Run basic functionality test:
```bash
python scripts/run_detection.py --image assets/test.jpg
```

For comprehensive testing, see the main project tests:
```bash
pytest tests/
```
