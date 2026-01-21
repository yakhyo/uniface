# Interactive Notebooks

Run UniFace examples directly in your browser with Google Colab, or download and run locally with Jupyter.

---

## Available Notebooks

| Notebook | Colab | Description |
|----------|:-----:|-------------|
| [Face Detection](https://github.com/yakhyo/uniface/blob/main/examples/01_face_detection.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yakhyo/uniface/blob/main/examples/01_face_detection.ipynb) | Detect faces and 5-point landmarks |
| [Face Alignment](https://github.com/yakhyo/uniface/blob/main/examples/02_face_alignment.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yakhyo/uniface/blob/main/examples/02_face_alignment.ipynb) | Align faces for recognition |
| [Face Verification](https://github.com/yakhyo/uniface/blob/main/examples/03_face_verification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yakhyo/uniface/blob/main/examples/03_face_verification.ipynb) | Compare faces for identity |
| [Face Search](https://github.com/yakhyo/uniface/blob/main/examples/04_face_search.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yakhyo/uniface/blob/main/examples/04_face_search.ipynb) | Find a person in group photos |
| [Face Analyzer](https://github.com/yakhyo/uniface/blob/main/examples/05_face_analyzer.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yakhyo/uniface/blob/main/examples/05_face_analyzer.ipynb) | All-in-one face analysis |
| [Face Parsing](https://github.com/yakhyo/uniface/blob/main/examples/06_face_parsing.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yakhyo/uniface/blob/main/examples/06_face_parsing.ipynb) | Semantic face segmentation |
| [Face Anonymization](https://github.com/yakhyo/uniface/blob/main/examples/07_face_anonymization.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yakhyo/uniface/blob/main/examples/07_face_anonymization.ipynb) | Privacy-preserving blur |
| [Gaze Estimation](https://github.com/yakhyo/uniface/blob/main/examples/08_gaze_estimation.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yakhyo/uniface/blob/main/examples/08_gaze_estimation.ipynb) | Gaze direction estimation |
| [Face Segmentation](https://github.com/yakhyo/uniface/blob/main/examples/09_face_segmentation.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yakhyo/uniface/blob/main/examples/09_face_segmentation.ipynb) | Face segmentation with XSeg |

---

## Running Locally

Download and run notebooks on your machine:

```bash
# Clone the repository
git clone https://github.com/yakhyo/uniface.git
cd uniface

# Install dependencies
pip install uniface jupyter

# Launch Jupyter
jupyter notebook examples/
```

---

## Running on Google Colab

Click any **"Open in Colab"** badge above. The notebooks automatically:

1. Install UniFace via pip
2. Clone the repository to access test images
3. Set up the correct working directory

!!! tip "GPU Acceleration"
    In Colab, go to **Runtime → Change runtime type → GPU** for faster inference.

---

## Next Steps

- [Quickstart](quickstart.md) - Code snippets for common use cases
- [Tutorials](recipes/image-pipeline.md) - Step-by-step workflow guides
- [API Reference](modules/detection.md) - Detailed module documentation
