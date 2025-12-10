# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UniFace is a production-ready face analysis library built on ONNX Runtime. It provides face detection, recognition, landmark detection, and attribute analysis (age, gender, emotion) with hardware acceleration support.

## Common Commands

```bash
# Install for development
uv pip install -e ".[dev]"

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_retinaface.py -v

# Run with coverage
uv run pytest --cov=uniface --cov-report=html

# Lint and format code
uv run ruff check .           # Check for linting errors
uv run ruff check . --fix     # Auto-fix linting errors
uv run ruff format .          # Format code

# Run a script
uv run python scripts/face_detection_demo.py images/image1.png
```

## Code Architecture

### Module Structure

- **uniface/detection/**: Face detectors (RetinaFace, SCRFD, YOLOv5Face) - all inherit from `BaseDetector`
- **uniface/recognition/**: Face encoders (ArcFace, MobileFace, SphereFace) - all inherit from `BaseRecognizer`
- **uniface/landmark/**: 106-point landmark detection
- **uniface/attribute/**: Age, gender, emotion prediction

### Key Design Patterns

1. **Abstract Base Classes**: `BaseDetector` (detection/base.py) and `BaseRecognizer` (recognition/base.py) define the interface. All models implement `detect()`, `preprocess()`, `postprocess()` for detectors and `get_embedding()`, `get_normalized_embedding()` for recognizers.

2. **Weight Enums**: Model weights are defined as Enums in `constants.py` (e.g., `RetinaFaceWeights.MNET_V2`). These map to download URLs and SHA-256 hashes.

3. **Model Store**: `model_store.py` handles automatic model downloading to `~/.uniface/models/` with SHA-256 verification via `verify_model_weights()`.

4. **ONNX Session Management**: `onnx_utils.py` provides `create_onnx_session()` which auto-selects the best execution provider (CUDA, CoreML, CPU).

5. **FaceAnalyzer**: High-level orchestrator in `analyzer.py` that combines detector + recognizer + attributes into a unified pipeline, returning `Face` dataclass objects.

### Detection Output Format

All detectors return `List[Dict]` where each dict contains:
- `bbox`: numpy array [x1, y1, x2, y2]
- `confidence`: float 0.0-1.0
- `landmarks`: numpy array (5, 2) for 5-point landmarks

## Code Style

- Line length: 120 characters
- Python target: 3.10+
- Quotes: Single quotes (configured in ruff)
- Imports: `uniface` is first-party (isort configured)

## Adding New Models

1. Create weight enum in `constants.py` with URL and SHA-256 hash
2. Implement model class inheriting from appropriate base class
3. Export from module `__init__.py` and main `uniface/__init__.py`
4. Add tests in `tests/`

## Git Branching Strategy

### Rules
1. **NEVER commit directly to `main`** - always use feature branches
2. Create branches from the latest `main`
3. Use PR workflow for all changes
4. Keep branches focused and short-lived

### Branch Naming Convention
- `feature/<name>` - New features (e.g., `feature/batch-processing`)
- `fix/<name>` - Bug fixes (e.g., `fix/memory-leak`)
- `docs/<name>` - Documentation updates (e.g., `docs/api-reference`)
- `perf/<name>` - Performance improvements (e.g., `perf/mlx-compile`)
- `refactor/<name>` - Code refactoring (e.g., `refactor/detection-base`)
- `test/<name>` - Test additions/changes (e.g., `test/recognition-coverage`)
- `ci/<name>` - CI/CD changes (e.g., `ci/github-actions`)

### Workflow
```bash
# 1. Start from latest main
git checkout main && git pull origin main

# 2. Create a feature branch
git checkout -b feature/my-feature

# 3. Make changes and commit
git add <files>
git commit -m "feat: add new feature"

# 4. Push branch and create PR
git push -u origin feature/my-feature
gh pr create --title "feat: Add new feature" --body "Description of changes"
```

### Commit Message Prefixes
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `perf:` - Performance improvement
- `refactor:` - Code refactoring (no functional change)
- `test:` - Adding or updating tests
- `ci:` - CI/CD configuration
- `chore:` - Maintenance tasks
