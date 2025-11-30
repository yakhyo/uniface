# Scripts

Scripts for testing UniFace features.

## Available Scripts

| Script | Description |
|--------|-------------|
| `run_detection.py` | Face detection on image or webcam |
| `run_age_gender.py` | Age and gender prediction |
| `run_emotion.py` | Emotion detection (7 or 8 emotions) |
| `run_landmarks.py` | 106-point facial landmark detection |
| `run_recognition.py` | Face embedding extraction and comparison |
| `run_face_analyzer.py` | Complete face analysis (detection + recognition + attributes) |
| `run_face_search.py` | Real-time face matching against reference |
| `run_video_detection.py` | Face detection on video files |
| `batch_process.py` | Batch process folder of images |
| `download_model.py` | Download model weights |
| `sha256_generate.py` | Generate SHA256 hash for model files |

## Usage Examples

```bash
# Face detection
python scripts/run_detection.py --image assets/test.jpg
python scripts/run_detection.py --webcam

# Age and gender
python scripts/run_age_gender.py --image assets/test.jpg
python scripts/run_age_gender.py --webcam

# Emotion detection
python scripts/run_emotion.py --image assets/test.jpg
python scripts/run_emotion.py --webcam

# Landmarks
python scripts/run_landmarks.py --image assets/test.jpg
python scripts/run_landmarks.py --webcam

# Face recognition (extract embedding)
python scripts/run_recognition.py --image assets/test.jpg

# Face comparison
python scripts/run_recognition.py --image1 face1.jpg --image2 face2.jpg

# Face search (match webcam against reference)
python scripts/run_face_search.py --image reference.jpg

# Video processing
python scripts/run_video_detection.py --input video.mp4 --output output.mp4

# Batch processing
python scripts/batch_process.py --input images/ --output results/

# Download models
python scripts/download_model.py --model-type retinaface
python scripts/download_model.py  # downloads all
```

## Common Options

| Option | Description |
|--------|-------------|
| `--image` | Path to input image |
| `--webcam` | Use webcam instead of image |
| `--detector` | Choose detector: `retinaface` or `scrfd` |
| `--threshold` | Visualization confidence threshold (default: 0.6) |
| `--save_dir` | Output directory (default: `outputs`) |

## Quick Test

```bash
python scripts/run_detection.py --image assets/test.jpg
```
