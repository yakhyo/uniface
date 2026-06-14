---
hide:
  - toc
  - navigation
  - edit
template: home.html
---

<div class="hero" markdown>

# UniFace { .hero-title }

<p class="hero-subtitle">A Unified Face Analysis Library for Python</p>

[![PyPI Version](https://img.shields.io/pypi/v/uniface.svg?label=Version)](https://pypi.org/project/uniface/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/uniface?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=Downloads)](https://pepy.tech/projects/uniface)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Github Build Status](https://github.com/yakhyo/uniface/actions/workflows/ci.yml/badge.svg)](https://github.com/yakhyo/uniface/actions)

[:material-rocket-launch: Get Started](quickstart.md){ .md-button .md-button--primary }
[:material-github: View on GitHub](https://github.com/yakhyo/uniface){ .md-button }

</div>

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
### :material-face-recognition: Face Detection
RetinaFace, SCRFD, and YOLO detectors with 5-point landmarks.
<a class="feature-card-link" href="modules/detection/" aria-label="Face Detection"></a>
</div>

<div class="feature-card" markdown>
### :material-account-check: Face Recognition
AdaFace, ArcFace, EdgeFace, MobileFace, and SphereFace embeddings for identity verification.
<a class="feature-card-link" href="modules/recognition/" aria-label="Face Recognition"></a>
</div>

<div class="feature-card" markdown>
### :material-map-marker: Landmarks
Dense facial landmark localization — 106-point (2d106det) and 98 / 68-point (PIPNet) variants.
<a class="feature-card-link" href="modules/landmarks/" aria-label="Landmarks"></a>
</div>

<div class="feature-card" markdown>
### :material-account-details: Attributes
Age, gender, race (FairFace), and emotion detection from faces.
<a class="feature-card-link" href="modules/attributes/" aria-label="Attributes"></a>
</div>

<div class="feature-card" markdown>
### :material-face-man-shimmer: Face Parsing
BiSeNet semantic segmentation with 19 facial component classes.
<a class="feature-card-link" href="modules/parsing/" aria-label="Face Parsing"></a>
</div>

<div class="feature-card" markdown>
### :material-eye: Gaze Estimation
Real-time gaze direction prediction with MobileGaze models.
<a class="feature-card-link" href="modules/gaze/" aria-label="Gaze Estimation"></a>
</div>

<div class="feature-card" markdown>
### :material-axis-arrow: Head Pose
3D head orientation (pitch, yaw, roll) estimation with 6D rotation models.
<a class="feature-card-link" href="modules/headpose/" aria-label="Head Pose"></a>
</div>

<div class="feature-card" markdown>
### :material-motion-play: Tracking
Multi-object tracking with BYTETracker for persistent face IDs across video frames.
<a class="feature-card-link" href="modules/tracking/" aria-label="Tracking"></a>
</div>

<div class="feature-card" markdown>
### :material-shield-check: Anti-Spoofing
Face liveness detection with MiniFASNet to prevent fraud.
<a class="feature-card-link" href="modules/spoofing/" aria-label="Anti-Spoofing"></a>
</div>

<div class="feature-card" markdown>
### :material-star-check: Face Quality
eDifFIQA scalar quality score to filter or rank faces before recognition.
<a class="feature-card-link" href="modules/quality/" aria-label="Face Quality"></a>
</div>

<div class="feature-card" markdown>
### :material-blur: Privacy
Face anonymization with 5 blur methods for privacy protection.
<a class="feature-card-link" href="modules/privacy/" aria-label="Privacy"></a>
</div>

<div class="feature-card" markdown>
### :material-database-search: Vector Indexing
FAISS-backed embedding store for fast multi-identity face search.
<a class="feature-card-link" href="modules/stores/" aria-label="Vector Indexing"></a>
</div>

</div>

---

## Installation

UniFace runs out of the box on macOS, Linux, and Windows, with automatic hardware acceleration on Apple Silicon, NVIDIA CUDA, and CPU. Inference runs on **ONNX Runtime** (with **PyTorch** for a few optional models) — installed automatically by the extra you choose below.

**CPU / Apple Silicon**
```bash
pip install uniface[cpu]
```

**GPU (NVIDIA CUDA)**
```bash
pip install uniface[gpu]
```

**From Source**
```bash
git clone https://github.com/yakhyo/uniface.git
cd uniface
pip install -e ".[cpu]"   # or .[gpu] for CUDA
```

---

## Next Steps

<div class="next-steps-grid" markdown>

<div class="feature-card" markdown>
### :material-rocket-launch: Quickstart
Get up and running in 5 minutes with common use cases.
<a class="feature-card-link" href="quickstart/" aria-label="Quickstart"></a>
</div>

<div class="feature-card" markdown>
### :material-school: Tutorials
Step-by-step examples for common workflows.
<a class="feature-card-link" href="recipes/image-pipeline/" aria-label="Tutorials"></a>
</div>

<div class="feature-card" markdown>
### :material-api: API Reference
Explore individual modules and their APIs.
<a class="feature-card-link" href="modules/detection/" aria-label="API Reference"></a>
</div>

<div class="feature-card" markdown>
### :material-book-open-variant: Guides
Learn about the architecture and design principles.
<a class="feature-card-link" href="concepts/overview/" aria-label="Guides"></a>
</div>

</div>

---

## License

UniFace is released under the [MIT License](https://opensource.org/licenses/MIT).
