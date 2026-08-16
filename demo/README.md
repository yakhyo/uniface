---
title: UniFace Demo
emoji: 🧑
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.46.0
app_file: app.py
license: mit
pinned: false
---

# UniFace Demo

Interactive demo for [UniFace](https://github.com/yakhyo/uniface) — a comprehensive library for face analysis built on ONNX Runtime.

## Features

- **Face Detection** — RetinaFace, SCRFD, YOLOv5-Face, YOLOv8-Face
- **Face Verification** — ArcFace, AdaFace, EdgeFace, MobileFace, SphereFace
- **Face Analysis** — Age, gender, and race prediction (AgeGender, FairFace)
- **Landmarks** — 106-point (2d106det), 98-point and 68-point (PIPNet)
- **Face Mesh** — Dense 468-point mesh, or 478-point with irises
- **Face Parsing** — BiSeNet (19 classes) and XSeg (face mask)
- **Emotion** — Expression recognition (AffectNet-7 / AffectNet-8)
- **Face States** — Eyes open, eyeglasses, sunglasses, and mask detection
- **Face Quality** — Recognition-suitability scoring with eDifFIQA
- **Gaze Estimation** — Pitch/yaw gaze direction (MobileGaze backbones)
- **Head Pose** — Pitch/yaw/roll with 3D cube or axis visualization
- **Portrait Matting** — Trimap-free MODNet alpha matte and background swap
- **Face Tracking** — ByteTrack multi-face tracking on video input
- **Anti-Spoofing** — Liveness detection (real vs. fake)
- **Face Anonymization** — Privacy-preserving blur methods

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Model weights download on first use and are cached afterwards, so the first run of
each tab is slower than the rest.

## Notes

- `torch` is required only by the Emotion tab, which loads a TorchScript model.
  Every other tab runs on ONNX Runtime alone.
- Example images live in `assets/`, grouped by tab.
