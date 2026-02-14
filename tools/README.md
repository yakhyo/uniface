# Tools

CLI utilities for testing and running UniFace features.

## Available Tools

| Tool | Description |
|------|-------------|
| `detect.py` | Face detection on image, video, or webcam |
| `track.py` | Face tracking on video with ByteTrack |
| `analyze.py` | Complete face analysis (detection + recognition + attributes) |
| `anonymize.py` | Face anonymization/blurring for privacy |
| `emotion.py` | Emotion detection (7 or 8 emotions) |
| `gaze.py` | Gaze direction estimation |
| `landmarks.py` | 106-point facial landmark detection |
| `recognize.py` | Face embedding extraction and comparison |
| `search.py` | Real-time face matching against reference |
| `fairface.py` | FairFace attribute prediction (race, gender, age) |
| `attribute.py` | Age and gender prediction |
| `spoofing.py` | Face anti-spoofing detection |
| `parse.py` | Face semantic segmentation (BiSeNet) |
| `xseg.py` | Face segmentation (XSeg) |
| `batch_process.py` | Batch process folder of images |
| `download_model.py` | Download model weights |
| `sha256_generate.py` | Generate SHA256 hash for model files |

## Unified `--source` Pattern

Most tools use a unified `--source` argument that accepts:
- **Image path**: `--source photo.jpg`
- **Video path**: `--source video.mp4`
- **Camera ID**: `--source 0` (default webcam), `--source 1` (external camera)

## Usage Examples

```bash
# Face detection
python tools/detect.py --source assets/test.jpg           # image
python tools/detect.py --source video.mp4                  # video
python tools/detect.py --source 0                          # webcam

# Face tracking
python tools/track.py --source video.mp4
python tools/track.py --source video.mp4 --output tracked.mp4
python tools/track.py --source 0                           # webcam

# Face anonymization
python tools/anonymize.py --source assets/test.jpg --method pixelate
python tools/anonymize.py --source video.mp4 --method gaussian
python tools/anonymize.py --source 0 --method pixelate

# Age and gender
python tools/attribute.py --source assets/test.jpg
python tools/attribute.py --source 0

# Emotion detection
python tools/emotion.py --source assets/test.jpg
python tools/emotion.py --source 0

# Gaze estimation
python tools/gaze.py --source assets/test.jpg
python tools/gaze.py --source 0

# Landmarks
python tools/landmarks.py --source assets/test.jpg
python tools/landmarks.py --source 0

# FairFace attributes
python tools/fairface.py --source assets/test.jpg
python tools/fairface.py --source 0

# Face parsing (BiSeNet)
python tools/parse.py --source assets/test.jpg
python tools/parse.py --source 0

# Face segmentation (XSeg)
python tools/xseg.py --source assets/test.jpg
python tools/xseg.py --source 0

# Face anti-spoofing
python tools/spoofing.py --source assets/test.jpg
python tools/spoofing.py --source 0

# Face analyzer
python tools/analyze.py --source assets/test.jpg
python tools/analyze.py --source 0

# Face recognition (extract embedding)
python tools/recognize.py --image assets/test.jpg

# Face comparison
python tools/recognize.py --image1 face1.jpg --image2 face2.jpg

# Face search (match against reference)
python tools/search.py --reference person.jpg --source 0
python tools/search.py --reference person.jpg --source video.mp4

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
python tools/detect.py --source assets/test.jpg
```
