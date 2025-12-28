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
5. Ensure all tests pass
6. Submit a pull request with a clear description

### Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check for linting errors
ruff check .

# Auto-fix linting errors
ruff check . --fix

# Format code
ruff format .
```

**Guidelines:**
- Follow PEP8 guidelines
- Use type hints (Python 3.10+)
- Write docstrings for public APIs
- Line length: 120 characters
- Keep code simple and readable

All PRs must pass `ruff check .` before merging.

## Development Setup

```bash
git clone https://github.com/yakhyo/uniface.git
cd uniface
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/
```

## Examples

Example notebooks demonstrating library usage:

| Example | Notebook |
|---------|----------|
| Face Detection | [01_face_detection.ipynb](examples/01_face_detection.ipynb) |
| Face Alignment | [02_face_alignment.ipynb](examples/02_face_alignment.ipynb) |
| Face Verification | [03_face_verification.ipynb](examples/03_face_verification.ipynb) |
| Face Search | [04_face_search.ipynb](examples/04_face_search.ipynb) |
| Face Analyzer | [05_face_analyzer.ipynb](examples/05_face_analyzer.ipynb) |
| Face Parsing | [06_face_parsing.ipynb](examples/06_face_parsing.ipynb) |
| Face Anonymization | [07_face_anonymization.ipynb](examples/07_face_anonymization.ipynb) |
| Gaze Estimation | [08_gaze_estimation.ipynb](examples/08_gaze_estimation.ipynb) |

## Questions?

Open an issue or start a discussion on GitHub.






