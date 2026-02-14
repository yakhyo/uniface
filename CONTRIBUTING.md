# Contributing to UniFace

Thank you for considering contributing to UniFace! We welcome contributions of all kinds.

## How to Contribute

### Reporting Issues

- Use GitHub Issues to report bugs or suggest features
- Include clear descriptions and reproducible examples
- Check existing issues before creating new ones

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature
3. Write clear, documented code with type hints
4. Add tests for new functionality
5. Ensure all tests pass and pre-commit hooks are satisfied
6. Submit a pull request with a clear description

## Development Setup

```bash
git clone https://github.com/yakhyo/uniface.git
cd uniface
pip install -e ".[dev]"
```

### Setting Up Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to ensure code quality and consistency. Install and configure it:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# (Optional) Run against all files
pre-commit run --all-files
```

Once installed, pre-commit will automatically run on every commit to check:

- Code formatting and linting (Ruff)
- Security issues (Bandit)
- General file hygiene (trailing whitespace, YAML/TOML validity, etc.)

**Note:** All PRs are automatically checked by CI. The merge button will only be available after all checks pass.

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, following modern Python best practices. Pre-commit handles all formatting automatically.

### Style Guidelines

#### General Rules

- **Line length:** 120 characters maximum
- **Python version:** 3.10+ (use modern syntax)
- **Quote style:** Single quotes for strings, double quotes for docstrings

#### Type Hints

Use modern Python 3.10+ type hints (PEP 585 and PEP 604):

```python
# Preferred (modern)
def process(items: list[str], config: dict[str, int] | None = None) -> tuple[int, str]:
    ...

# Avoid (legacy)
from typing import List, Dict, Optional, Tuple
def process(items: List[str], config: Optional[Dict[str, int]] = None) -> Tuple[int, str]:
    ...
```

#### Docstrings

Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for all public APIs:

```python
def create_detector(method: str = 'retinaface', **kwargs: Any) -> BaseDetector:
    """Factory function to create face detectors.

    Args:
        method: Detection method. Options: 'retinaface', 'scrfd', 'yolov5face', 'yolov8face'.
        **kwargs: Detector-specific parameters.

    Returns:
        Initialized detector instance.

    Raises:
        ValueError: If method is not supported.

    Example:
        >>> from uniface import create_detector
        >>> detector = create_detector('retinaface', confidence_threshold=0.8)
        >>> faces = detector.detect(image)
        >>> print(f"Found {len(faces)} faces")
    """
```

#### Import Order

Imports are automatically sorted by Ruff with the following order:

1. **Future** imports (`from __future__ import annotations`)
2. **Standard library** (`os`, `sys`, `typing`, etc.)
3. **Third-party** (`numpy`, `cv2`, `onnxruntime`, etc.)
4. **First-party** (`uniface.*`)
5. **Local** (relative imports like `.base`, `.models`)

```python
from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from uniface.constants import RetinaFaceWeights
from uniface.log import Logger

from .base import BaseDetector
```

#### Code Comments

- Add comments for complex logic, magic numbers, and non-obvious behavior
- Avoid comments that merely restate the code
- Use `# TODO:` with issue links for planned improvements

```python
# RetinaFace FPN strides and corresponding anchor sizes per level
steps = [8, 16, 32]
min_sizes = [[16, 32], [64, 128], [256, 512]]

# Add small epsilon to prevent division by zero
similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-5)
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_factory.py

# Run with coverage
pytest tests/ --cov=uniface --cov-report=html
```

## Adding New Features

When adding a new model or feature:

1. **Create the model class** in the appropriate submodule (e.g., `uniface/detection/`)
2. **Add weight constants** to `uniface/constants.py` with URLs and SHA256 hashes
3. **Export in `__init__.py`** files at both module and package levels
4. **Write tests** in `tests/` directory
5. **Add example usage** in `tools/` or update existing notebooks
6. **Update documentation** if needed

## Examples

Example notebooks demonstrating library usage:

| Example            | Notebook                                                            |
| ------------------ | ------------------------------------------------------------------- |
| Face Detection     | [01_face_detection.ipynb](examples/01_face_detection.ipynb)         |
| Face Alignment     | [02_face_alignment.ipynb](examples/02_face_alignment.ipynb)         |
| Face Verification  | [03_face_verification.ipynb](examples/03_face_verification.ipynb)   |
| Face Search        | [04_face_search.ipynb](examples/04_face_search.ipynb)               |
| Face Analyzer      | [05_face_analyzer.ipynb](examples/05_face_analyzer.ipynb)           |
| Face Parsing       | [06_face_parsing.ipynb](examples/06_face_parsing.ipynb)             |
| Face Anonymization | [07_face_anonymization.ipynb](examples/07_face_anonymization.ipynb) |
| Gaze Estimation    | [08_gaze_estimation.ipynb](examples/08_gaze_estimation.ipynb)       |

## Questions?

Open an issue or start a discussion on GitHub.
