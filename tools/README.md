# Tools

CLI utilities for testing and running UniFace features.

## Available Tools

| Tool | Description |
|------|-------------|
| `detection.py` | Face detection on image, video, or webcam |
| `face_anonymize.py` | Face anonymization/blurring for privacy |
| `age_gender.py` | Age and gender prediction |
| `face_emotion.py` | Emotion detection (7 or 8 emotions) |
| `gaze_estimation.py` | Gaze direction estimation |
| `landmarks.py` | 106-point facial landmark detection |
| `recognition.py` | Face embedding extraction and comparison |
| `face_analyzer.py` | Complete face analysis (detection + recognition + attributes) |
| `face_search.py` | Real-time face matching against reference |
| `fairface.py` | FairFace attribute prediction (race, gender, age) |
| `spoofing.py` | Face anti-spoofing detection |
| `face_parsing.py` | Face semantic segmentation (BiSeNet) |
| `xseg.py` | Face segmentation (XSeg) |
| `video_detection.py` | Face detection on video files with progress bar |
| `batch_process.py` | Batch process folder of images |
| `download_model.py` | Download model weights |
| `sha256_generate.py` | Generate SHA256 hash for model files |

## Unified `--source` Pattern

All tools use a unified `--source` argument that accepts:
- **Image path**: `--source photo.jpg`
- **Video path**: `--source video.mp4`
- **Camera ID**: `--source 0` (default webcam), `--source 1` (external camera)

## Usage Examples

```bash
# Face detection
python tools/detection.py --source assets/test.jpg           # image
python tools/detection.py --source video.mp4                 # video
python tools/detection.py --source 0                         # webcam

# Face anonymization
python tools/face_anonymize.py --source assets/test.jpg --method pixelate
python tools/face_anonymize.py --source video.mp4 --method gaussian
python tools/face_anonymize.py --source 0 --method pixelate

# Age and gender
python tools/age_gender.py --source assets/test.jpg
python tools/age_gender.py --source 0

# Emotion detection
python tools/face_emotion.py --source assets/test.jpg
python tools/face_emotion.py --source 0

# Gaze estimation
python tools/gaze_estimation.py --source assets/test.jpg
python tools/gaze_estimation.py --source 0

# Landmarks
python tools/landmarks.py --source assets/test.jpg
python tools/landmarks.py --source 0

# FairFace attributes
python tools/fairface.py --source assets/test.jpg
python tools/fairface.py --source 0

# Face parsing (BiSeNet)
python tools/face_parsing.py --source assets/test.jpg
python tools/face_parsing.py --source 0

# Face segmentation (XSeg)
python tools/xseg.py --source assets/test.jpg
python tools/xseg.py --source 0

# Face anti-spoofing
python tools/spoofing.py --source assets/test.jpg
python tools/spoofing.py --source 0

# Face analyzer
python tools/face_analyzer.py --source assets/test.jpg
python tools/face_analyzer.py --source 0

# Face recognition (extract embedding)
python tools/recognition.py --image assets/test.jpg

# Face comparison
python tools/recognition.py --image1 face1.jpg --image2 face2.jpg

# Face search (match against reference)
python tools/face_search.py --reference person.jpg --source 0
python tools/face_search.py --reference person.jpg --source video.mp4

# Video processing with progress bar
python tools/video_detection.py --source video.mp4
python tools/video_detection.py --source video.mp4 --output output.mp4

# Batch processing
python tools/batch_process.py --input images/ --output results/

# Download models
python tools/download_model.py --model-type retinaface
python tools/download_model.py  # downloads all
```

## Common Options

| Option | Description |
|--------|-------------|
| `--source` | Input source: image/video path or camera ID (0, 1, ...) |
| `--detector` | Choose detector: `retinaface`, `scrfd`, `yolov5face` |
| `--threshold` | Visualization confidence threshold (default: varies) |
| `--save-dir` | Output directory (default: `outputs`) |

## Supported Formats

**Images:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff`

**Videos:** `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`

**Camera:** Use integer IDs (`0`, `1`, `2`, ...)

## Quick Test

```bash
python tools/detection.py --source assets/test.jpg
```
