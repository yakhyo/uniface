# Contributing

Thank you for contributing to UniFace!

---

## Quick Start

We use [uv](https://docs.astral.sh/uv/) for reproducible dev installs (lockfile-pinned).

```bash
# Install uv first: https://docs.astral.sh/uv/getting-started/installation/

# Clone
git clone https://github.com/yakhyo/uniface.git
cd uniface

# Install runtime + cpu + dev extras from uv.lock (--extra gpu for CUDA)
uv sync --extra cpu --extra dev

# Run tests
uv run pytest
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
- Python 3.10+ type hints
- Google-style docstrings

---

## Pre-commit Hooks

`pre-commit` is included in the `[dev]` extra, so `uv sync` already installs it.

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

---

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short description>
```

| Type         | When to use                                      |
|--------------|--------------------------------------------------|
| **feat**     | New feature or capability                        |
| **fix**      | Bug fix                                          |
| **docs**     | Documentation changes                            |
| **style**    | Formatting, whitespace (no logic change)         |
| **refactor** | Code restructuring without changing behavior     |
| **perf**     | Performance improvement                          |
| **test**     | Adding or updating tests                         |
| **build**    | Build system or dependencies                     |
| **ci**       | CI/CD and pre-commit configuration               |
| **chore**    | Routine maintenance and tooling                  |

**Examples:**

```
feat: Add gaze estimation model
fix: Correct bounding box scaling for non-square images
ci: Add nbstripout pre-commit hook
docs: Update installation instructions
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

## Releases

Releases are automated via GitHub Actions. Maintainers trigger **Actions → Release Pipeline → Run workflow** with a [PEP 440](https://peps.python.org/pep-0440/) version (e.g. `0.7.0`, `0.7.0rc1`). The pipeline runs tests, bumps `pyproject.toml` + `uniface/__init__.py`, tags the commit, publishes to PyPI, and creates a GitHub Release. Docs redeploy only for stable releases.

See [CONTRIBUTING.md](https://github.com/yakhyo/uniface/blob/main/CONTRIBUTING.md#release-process) for the full process.

---

## Questions?

Open an issue on [GitHub](https://github.com/yakhyo/uniface/issues).
