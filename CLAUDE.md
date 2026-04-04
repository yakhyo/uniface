# CLAUDE.md

Project instructions for AI coding agents.

## Project Overview

UniFace is a Python library for face detection, recognition, tracking, landmark analysis, face parsing, gaze estimation, age/gender detection. It uses ONNX Runtime for inference.

## Code Style

- Python 3.10+ with type hints
- Line length: 120
- Single quotes for strings, double quotes for docstrings
- Google-style docstrings
- Formatter/linter: Ruff (config in `pyproject.toml`)
- Run `ruff format .` and `ruff check . --fix` before committing

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/) with a **capitalized** description:

```
<type>: <Capitalized short description>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`

Examples:
- `feat: Add gaze estimation model`
- `fix: Correct bounding box scaling for non-square images`
- `ci: Add nbstripout pre-commit hook`
- `docs: Update installation instructions`
- `refactor: Unify attribute/detector base classes`

## Testing

```bash
pytest -v --tb=short
```

Tests live in `tests/`. Run the full suite before submitting changes.

## Pre-commit

Pre-commit hooks handle formatting, linting, security checks, and notebook output stripping. Always run:

```bash
pre-commit install
pre-commit run --all-files
```

## Project Structure

```
uniface/            # Main package
  detection/        # Face detection models (SCRFD, RetinaFace, YOLOv5, YOLOv8)
  recognition/      # Face recognition/verification (AdaFace, ArcFace, EdgeFace, MobileFace, SphereFace)
  landmark/         # Facial landmark models
  tracking/         # Object tracking (ByteTrack)
  parsing/          # Face parsing/segmentation (BiSeNet, XSeg)
  gaze/             # Gaze estimation
  headpose/         # Head pose estimation
  attribute/        # Age, gender, emotion detection
  spoofing/         # Anti-spoofing (MiniFASNet)
  privacy/          # Face anonymization
  stores/           # Vector stores (FAISS)
  constants.py      # Model weight URLs and checksums
  model_store.py    # Model download/cache management
  analyzer.py       # High-level FaceAnalyzer API
  types.py          # Shared type definitions
tests/              # Unit tests
examples/           # Jupyter notebooks (outputs are auto-stripped)
docs/               # MkDocs documentation
```

## Key Conventions

- New models: add class in submodule, register weights in `constants.py`, export in `__init__.py`
- Dependencies: managed in `pyproject.toml`
- All ONNX models are downloaded on demand with SHA256 verification
- Do not commit notebook outputs; `nbstripout` pre-commit hook handles this
