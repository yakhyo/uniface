---
hide:
  - toc
  - navigation
  - edit
template: home.html
---

<div class="hero" markdown>

# UniFace { .hero-title }

<p class="hero-subtitle">A lightweight, production-ready face analysis library built on ONNX Runtime</p>

[![PyPI](https://img.shields.io/pypi/v/uniface.svg)](https://pypi.org/project/uniface/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/uniface)](https://pepy.tech/project/uniface)

[Get Started](quickstart.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/yakhyo/uniface){ .md-button }

</div>

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
### :material-face-recognition: Face Detection
ONNX-optimized RetinaFace, SCRFD, and YOLOv5-Face models with 5-point landmarks.
</div>

<div class="feature-card" markdown>
### :material-account-check: Face Recognition
ArcFace, MobileFace, and SphereFace embeddings for identity verification.
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
### :material-shield-check: Anti-Spoofing
Face liveness detection with MiniFASNet to prevent fraud.
</div>

<div class="feature-card" markdown>
### :material-blur: Privacy
Face anonymization with 5 blur methods for privacy protection.
</div>

</div>

---

## Installation

=== "Standard"

    ```bash
    pip install uniface
    ```

=== "GPU (CUDA)"

    ```bash
    pip install uniface[gpu]
    ```

=== "From Source"

    ```bash
    git clone https://github.com/yakhyo/uniface.git
    cd uniface
    pip install -e .
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
### :material-book-open-variant: Concepts
Learn about the architecture and design principles.

[Read Concepts →](concepts/overview.md)
</div>

<div class="feature-card" markdown>
### :material-puzzle: Modules
Explore individual modules and their APIs.

[Browse Modules →](modules/detection.md)
</div>

<div class="feature-card" markdown>
### :material-chef-hat: Recipes
Complete examples for common workflows.

[View Recipes →](recipes/image-pipeline.md)
</div>

</div>

---

UniFace is released under the [MIT License](https://opensource.org/licenses/MIT).
