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

- Follow PEP8 guidelines
- Use type hints (Python 3.10+)
- Write docstrings for public APIs
- Keep code simple and readable

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
| Face Detection | [face_detection.ipynb](examples/face_detection.ipynb) |
| Face Alignment | [face_alignment.ipynb](examples/face_alignment.ipynb) |
| Face Recognition | [face_analyzer.ipynb](examples/face_analyzer.ipynb) |
| Face Verification | [face_verification.ipynb](examples/face_verification.ipynb) |
| Face Search | [face_search.ipynb](examples/face_search.ipynb) |

## Questions?

Open an issue or start a discussion on GitHub.

