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
| `headpose.py` | Head pose estimation (pitch, yaw, roll) |
| `landmarks.py` | 106-point facial landmark detection |
| `facemesh.py` | 468 / 478-point dense 3D face mesh (MediaPipe), irises with `--model v2_478` |
| `recognize.py` | Face embedding extraction and comparison |
| `search.py` | Real-time face matching against reference |
| `faiss_search.py` | FAISS index build and multi-identity face search |
| `fairface.py` | FairFace attribute prediction (race, gender, age) |
| `attribute.py` | Age and gender prediction |
| `facestate.py` | Face states (eye openness, glasses, mask, sunglasses) with FaceAttribNet |
| `spoofing.py` | Face anti-spoofing detection |
| `quality.py` | Face image quality assessment (eDifFIQA) |
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
python tools/detect.py --source assets/source/detect_group.jpg           # image
python tools/detect.py --source video.mp4                  # video
python tools/detect.py --source 0                          # webcam

# Face tracking
python tools/track.py --source video.mp4
python tools/track.py --source video.mp4 --output tracked.mp4
python tools/track.py --source 0                           # webcam

# Face anonymization
python tools/anonymize.py --source assets/source/anon_group.jpg --method pixelate
python tools/anonymize.py --source video.mp4 --method gaussian
python tools/anonymize.py --source 0 --method pixelate

# Age and gender
python tools/attribute.py --source assets/source/age_adult.jpg
python tools/attribute.py --source 0

# Emotion detection
python tools/emotion.py --source assets/source/emotion_happy.jpg
python tools/emotion.py --source 0

# Face states (eye openness, glasses, mask, sunglasses)
python tools/facestate.py --source assets/source/state_b_glasses.jpg
python tools/facestate.py --source 0
python tools/facestate.py --source assets/source/state_b_glasses.jpg --threshold 0.7 --margin 0.1

# Gaze estimation
python tools/gaze.py --source assets/source/gaze_averted.jpg
python tools/gaze.py --source 0

# Head pose estimation
python tools/headpose.py --source assets/source/pose_right.jpg
python tools/headpose.py --source 0
python tools/headpose.py --source 0 --draw-type axis

# Landmarks
python tools/landmarks.py --source assets/source/landmarks_face.jpg
python tools/landmarks.py --source 0

# Face mesh (468 / 478-point dense 3D)
python tools/facemesh.py --source assets/source/mesh_face.jpg
python tools/facemesh.py --source 0 --mode points
python tools/facemesh.py --source assets/source/mesh_face.jpg --detector blazeface  # MediaPipe parity
python tools/facemesh.py --source 0 --model v2_478 --mode points        # with irises

# FairFace attributes
python tools/fairface.py --source assets/source/age_adult.jpg
python tools/fairface.py --source 0

# Face parsing (BiSeNet)
python tools/parse.py --source assets/source/parse_face.jpg
python tools/parse.py --source 0

# Face segmentation (XSeg)
python tools/xseg.py --source assets/source/seg_face.jpg
python tools/xseg.py --source 0

# Face anti-spoofing
python tools/spoofing.py --source assets/source/age_adult.jpg
python tools/spoofing.py --source 0

# Face image quality assessment (eDifFIQA)
python tools/quality.py --source assets/source/detect_group.jpg
python tools/quality.py --source 0
python tools/quality.py --source assets/source/detect_group.jpg --variant l

# Face analyzer
python tools/analyze.py --source assets/source/detect_group.jpg
python tools/analyze.py --source 0

# Face recognition (extract embedding)
python tools/recognize.py --image assets/source/verify_now_2010.jpg

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
| `--detector` | Choose detector: `retinaface`, `scrfd`, `centerface`, `blazeface`, `yolov5face`, `yolov8face` |
| `--threshold` | Visualization confidence threshold (default: varies) |
| `--save-dir` | Output directory (default: `outputs`) |

## Supported Formats

**Images:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff`

**Videos:** `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`

**Camera:** Use integer IDs (`0`, `1`, `2`, ...)

## Quick Test

```bash
python tools/detect.py --source assets/source/detect_group.jpg
```
