---
title: UniFace Demo
emoji: 🧑
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.12.0
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
- **Landmarks** — 106-point facial landmark detection
- **Face Parsing** — BiSeNet (19 classes) and XSeg (face mask)
- **Gaze Estimation** — Pitch/yaw gaze direction (MobileGaze backbones)
- **Head Pose** — Pitch/yaw/roll with 3D cube or axis visualization
- **Portrait Matting** — Trimap-free MODNet alpha matte and background swap
- **Face Tracking** — ByteTrack multi-face tracking on video input
- **Anti-Spoofing** — Liveness detection (real vs. fake)
- **Face Anonymization** — Privacy-preserving blur methods
