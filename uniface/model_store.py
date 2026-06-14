# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
import hashlib
import os
import tempfile
import time

import requests
from tqdm import tqdm

import uniface.constants as const
from uniface.log import Logger

__all__ = ['download_models', 'get_cache_dir', 'set_cache_dir', 'verify_model_weights']

_DEFAULT_CACHE_DIR = '~/.uniface/models'
_ENV_KEY = 'UNIFACE_CACHE_DIR'


def get_cache_dir() -> str:
    """Get the current model cache directory path.

    Resolution order:
        1. ``UNIFACE_CACHE_DIR`` environment variable (set via :func:`set_cache_dir` or directly).
        2. Default: ``~/.uniface/models``.

    Returns:
        Absolute, expanded path to the cache directory.

    Example:
        >>> from uniface import get_cache_dir
        >>> print(get_cache_dir())
        '/home/user/.uniface/models'
    """
    return os.path.expanduser(os.environ.get(_ENV_KEY, _DEFAULT_CACHE_DIR))


def set_cache_dir(path: str) -> None:
    """Set the model cache directory.

    This sets the ``UNIFACE_CACHE_DIR`` environment variable so that all
    subsequent model downloads and lookups use the new path.

    Args:
        path: Directory path for storing model weights.

    Example:
        >>> from uniface import set_cache_dir, get_cache_dir
        >>> set_cache_dir('/data/models')
        >>> print(get_cache_dir())
        '/data/models'
    """
    os.environ[_ENV_KEY] = path
    Logger.info(f'Cache directory set to: {path}')


def verify_model_weights(
    model_name: Enum,
    root: str | None = None,
    timeout: int = 60,
    max_retries: int = 3,
) -> str:
    """Ensure model weights are present, downloading and verifying them if necessary.

    Given a model identifier from an Enum class (e.g., `RetinaFaceWeights.MNET_V2`),
    this function checks if the corresponding weight file exists locally. If not,
    it downloads the file from a predefined URL and verifies its integrity using
    a SHA-256 hash.

    Args:
        model_name: Model weight identifier enum (e.g., `RetinaFaceWeights.MNET_V2`).
        root: Directory to store or locate the model weights.
            If None, uses the cache directory from :func:`get_cache_dir`.
        timeout: Connection timeout in seconds. Defaults to 60.
        max_retries: Maximum number of download attempts. Defaults to 3.

    Returns:
        Absolute path to the verified model weights file.

    Raises:
        ValueError: If the model is unknown or SHA-256 verification fails.
        ConnectionError: If downloading the file fails.

    Example:
        >>> from uniface.constants import RetinaFaceWeights
        >>> from uniface.model_store import verify_model_weights
        >>> path = verify_model_weights(RetinaFaceWeights.MNET_V2)
        >>> print(path)
        '/home/user/.uniface/models/retinaface_mnet_v2.onnx'
    """

    root = os.path.expanduser(root) if root is not None else get_cache_dir()
    os.makedirs(root, exist_ok=True)

    # Lookup model info from registry
    model_info = const.MODEL_REGISTRY.get(model_name)
    if not model_info:
        Logger.error(f"No entry found in MODEL_REGISTRY for model '{model_name}'")
        raise ValueError(f"Unknown model identifier: '{model_name}'")

    url = model_info.url
    expected_hash = model_info.sha256

    file_ext = os.path.splitext(url)[1]
    model_path = os.path.normpath(os.path.join(root, f'{model_name.value}{file_ext}'))

    # Re-download if the cached file is missing or fails verification (e.g. corrupted externally).
    if os.path.exists(model_path) and expected_hash and not verify_file_hash(model_path, expected_hash):
        Logger.warning(f"Cached weights for '{model_name.value}' are corrupted; re-downloading.")
        os.remove(model_path)

    if not os.path.exists(model_path):
        Logger.info(f"Downloading model '{model_name.value}' from {url}")
        try:
            download_file(url, model_path, expected_hash=expected_hash, timeout=timeout, max_retries=max_retries)
            Logger.info(f"Successfully downloaded '{model_name.value}' to {model_path}")
        except Exception as e:
            Logger.error(f"Failed to download model '{model_name.value}': {e}")
            raise ConnectionError(f"Download failed for '{model_name.value}' after {max_retries} attempts") from e

    return model_path


def download_file(
    url: str,
    dest_path: str,
    expected_hash: str | None = None,
    timeout: int = 60,
    max_retries: int = 3,
) -> None:
    """Download a file with retries, streaming to a temp file and committing atomically.

    Bytes are written to a temp file, optionally hash-verified, then moved into
    place with :func:`os.replace`, so ``dest_path`` is never left partial or corrupted.

    Args:
        url: URL to download from.
        dest_path: Local file path to save to.
        expected_hash: Expected SHA-256 hash; if set, a mismatch triggers a retry.
        timeout: Connection timeout in seconds. Defaults to 60.
        max_retries: Maximum number of attempts. Defaults to 3.
    """
    last_error = None
    dest_dir = os.path.dirname(dest_path) or '.'
    for attempt in range(max_retries):
        tmp_path = None
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            # Write to a unique temp file in the same directory so os.replace is atomic.
            fd, tmp_path = tempfile.mkstemp(dir=dest_dir, suffix='.tmp')
            with (
                os.fdopen(fd, 'wb') as file,
                tqdm(
                    total=total_size,
                    desc=f'Attempt {attempt + 1}/{max_retries}',
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress,
            ):
                for chunk in response.iter_content(chunk_size=const.DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        file.write(chunk)
                        progress.update(len(chunk))

            if expected_hash and not verify_file_hash(tmp_path, expected_hash):
                raise ValueError('SHA-256 hash mismatch on downloaded file')

            os.replace(tmp_path, dest_path)  # Atomic commit
            return  # Success
        except (OSError, requests.RequestException, ValueError) as e:
            last_error = e
            Logger.warning(f'Download attempt {attempt + 1} failed: {e}. Retrying...')
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    raise ConnectionError(f'Failed to download file from {url}. Error: {last_error}')


def verify_file_hash(file_path: str, expected_hash: str) -> bool:
    """Compute the SHA-256 hash of the file and compare it with the expected hash."""
    file_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(const.HASH_CHUNK_SIZE), b''):
            file_hash.update(chunk)
    actual_hash = file_hash.hexdigest()
    if actual_hash != expected_hash:
        Logger.warning(f'Expected hash: {expected_hash}, but got: {actual_hash}')
    return actual_hash == expected_hash


def download_models(
    model_names: list[Enum], max_workers: int | None = None, timeout: int = 60, max_retries: int = 3
) -> dict[Enum, str]:
    """Download and verify multiple models concurrently.

    Uses a thread pool to download models in parallel, which is significantly
    faster when initializing several models at once.

    Args:
        model_names: List of model weight enum identifiers to download.
        max_workers: Maximum number of concurrent download threads. Defaults to
            ``min(os.cpu_count() or 1, 8)`` (auto mode). Passing ``None`` or a
            value ``< 1`` also selects auto mode and emits an info log line.
        timeout: Connection timeout in seconds. Defaults to 60.
        max_retries: Maximum number of attempts per model. Defaults to 3.

    Returns:
        Mapping of each model enum to its local file path.

    Raises:
        TypeError: If ``max_workers`` is a ``bool`` or a non-int / non-None
            value.
        RuntimeError: If any model download or verification fails. The error
            message aggregates every failure into a single multi-line message
            of the form ``"Failed to download N model(s):\\n<name>: <err>\\n..."``.

    Example:
        >>> from uniface import download_models
        >>> from uniface.constants import RetinaFaceWeights, ArcFaceWeights
        >>> paths = download_models([RetinaFaceWeights.MNET_V2, ArcFaceWeights.RESNET])
    """
    results: dict[Enum, str] = {}
    errors: list[str] = []

    if isinstance(max_workers, bool) or not isinstance(max_workers, int | None):
        raise TypeError(f'max_workers must be int or None, got {type(max_workers).__name__}')

    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, 8)
    elif max_workers < 1:
        Logger.info(f'max_workers must be >= 1, got {max_workers}; falling back to auto mode')
        max_workers = min(os.cpu_count() or 1, 8)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_model = {
            executor.submit(verify_model_weights, name, timeout=timeout, max_retries=max_retries): name
            for name in model_names
        }

        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                path = future.result()
                results[model] = path
                Logger.info(f'Ready: {model.value} -> {path}')
            except Exception as e:
                errors.append(f'{model.value}: {e}')
                Logger.error(f'Failed to download {model.value}: {e}')

    if errors:
        raise RuntimeError(f'Failed to download {len(errors)} model(s):\n' + '\n'.join(errors))

    Logger.info(f'All {len(results)} model(s) downloaded and verified')
    return results
