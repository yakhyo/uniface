# Contributing

Thank you for contributing to UniFace!

---

## Quick Start

```bash
# Clone
git clone https://github.com/yakhyo/uniface.git
cd uniface

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

---

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for formatting:

```bash
ruff format .
ruff check . --fix
```

**Guidelines:**

- Line length: 120
- Python 3.11+ type hints
- Google-style docstrings

---

## Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

---

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Write tests for new features
4. Ensure tests pass
5. Submit PR with clear description

---

## Adding New Models

1. Create model class in appropriate submodule
2. Add weight constants to `uniface/constants.py`
3. Export in `__init__.py` files
4. Write tests in `tests/`
5. Add example in `tools/` or notebooks

---

## Questions?

Open an issue on [GitHub](https://github.com/yakhyo/uniface/issues).
