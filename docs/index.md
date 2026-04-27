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
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Github Build Status](https://github.com/yakhyo/uniface/actions/workflows/ci.yml/badge.svg)](https://github.com/yakhyo/uniface/actions)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/uniface?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=Downloads)](https://pepy.tech/projects/uniface)
[![Kaggle Badge](https://img.shields.io/badge/Notebooks-Kaggle?label=Kaggle&color=blue)](https://www.kaggle.com/yakhyokhuja/code)
[![Discord](https://img.shields.io/badge/Discord-Join%20Server-5865F2?logo=discord&logoColor=white)](https://discord.gg/wdzrjr7R5j)

<!-- <img src="https://raw.githubusercontent.com/yakhyo/uniface/main/.github/logos/uniface_rounded_q80.webp" alt="UniFace - A Unified Face Analysis Library for Python" style="max-width: 70%; margin: 1rem 0;"> -->

[Get Started](quickstart.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/yakhyo/uniface){ .md-button }

</div>

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
### :material-face-recognition: Face Detection
RetinaFace, SCRFD, and YOLO detectors with 5-point landmarks.
</div>

<div class="feature-card" markdown>
### :material-account-check: Face Recognition
AdaFace, ArcFace, EdgeFace, MobileFace, and SphereFace embeddings for identity verification.
</div>

<div class="feature-card" markdown>
### :material-map-marker: Landmarks
Accurate 106-point facial landmark localization for detailed face analysis.
</div>

<div class="feature-card" markdown>
### :material-account-details: Attributes
Age, gender, race (FairFace), and emotion detection from faces.
</div>

<div class="feature-card" markdown>
### :material-face-man-shimmer: Face Parsing
BiSeNet semantic segmentation with 19 facial component classes.
</div>

<div class="feature-card" markdown>
### :material-eye: Gaze Estimation
Real-time gaze direction prediction with MobileGaze models.
</div>

<div class="feature-card" markdown>
### :material-axis-arrow: Head Pose
3D head orientation (pitch, yaw, roll) estimation with 6D rotation models.
</div>

<div class="feature-card" markdown>
### :material-motion-play: Tracking
Multi-object tracking with BYTETracker for persistent face IDs across video frames.
</div>

<div class="feature-card" markdown>
### :material-shield-check: Anti-Spoofing
Face liveness detection with MiniFASNet to prevent fraud.
</div>

<div class="feature-card" markdown>
### :material-blur: Privacy
Face anonymization with 5 blur methods for privacy protection.
</div>

<div class="feature-card" markdown>
### :material-database-search: Vector Indexing
FAISS-backed embedding store for fast multi-identity face search.
</div>

</div>

---

## Installation

UniFace uses portable model runtimes for consistent inference across macOS, Linux, and Windows. Most core components run through **ONNX Runtime**, while optional components may use **PyTorch** where appropriate.

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

[Quickstart Guide →](quickstart.md)
</div>

<div class="feature-card" markdown>
### :material-school: Tutorials
Step-by-step examples for common workflows.

[View Tutorials →](recipes/image-pipeline.md)
</div>

<div class="feature-card" markdown>
### :material-api: API Reference
Explore individual modules and their APIs.

[Browse API →](modules/detection.md)
</div>

<div class="feature-card" markdown>
### :material-book-open-variant: Guides
Learn about the architecture and design principles.

[Read Guides →](concepts/overview.md)
</div>

</div>

---

## License

UniFace is released under the [MIT License](https://opensource.org/licenses/MIT).
