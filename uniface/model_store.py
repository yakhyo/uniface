# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Model weight management for UniFace.

This module handles downloading, caching, and verifying model weights
using SHA-256 checksums for integrity validation.
"""

from __future__ import annotations

from enum import Enum
import hashlib
import os

import requests
from tqdm import tqdm

import uniface.constants as const
from uniface.log import Logger

__all__ = ['verify_model_weights']


def verify_model_weights(model_name: Enum, root: str = '~/.uniface/models') -> str:
    """Ensure model weights are present, downloading and verifying them if necessary.

    Given a model identifier from an Enum class (e.g., `RetinaFaceWeights.MNET_V2`),
    this function checks if the corresponding weight file exists locally. If not,
    it downloads the file from a predefined URL and verifies its integrity using
    a SHA-256 hash.

    Args:
        model_name: Model weight identifier enum (e.g., `RetinaFaceWeights.MNET_V2`).
        root: Directory to store or locate the model weights.
            Defaults to '~/.uniface/models'.

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

    root = os.path.expanduser(root)
    os.makedirs(root, exist_ok=True)

    # Keep model_name as enum for dictionary lookup
    url = const.MODEL_URLS.get(model_name)
    if not url:
        Logger.error(f"No URL found for model '{model_name}'")
        raise ValueError(f"No URL found for model '{model_name}'")

    file_ext = os.path.splitext(url)[1]
    model_path = os.path.normpath(os.path.join(root, f'{model_name.value}{file_ext}'))

    if not os.path.exists(model_path):
        Logger.info(f"Downloading model '{model_name}' from {url}")
        try:
            download_file(url, model_path)
            Logger.info(f"Successfully downloaded '{model_name}' to {model_path}")
        except Exception as e:
            Logger.error(f"Failed to download model '{model_name}': {e}")
            raise ConnectionError(f"Download failed for '{model_name}'") from e

    expected_hash = const.MODEL_SHA256.get(model_name)
    if expected_hash and not verify_file_hash(model_path, expected_hash):
        os.remove(model_path)  # Remove corrupted file
        Logger.warning('Corrupted weight detected. Removing...')
        raise ValueError(f"Hash mismatch for '{model_name}'. The file may be corrupted; please try downloading again.")

    return model_path


def download_file(url: str, dest_path: str, timeout: int = 30) -> None:
    """Download a file from a URL in chunks and save it to the destination path.

    Args:
        url: URL to download from.
        dest_path: Local file path to save to.
        timeout: Connection timeout in seconds. Defaults to 30.
    """
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        with (
            open(dest_path, 'wb') as file,
            tqdm(
                desc=f'Downloading {dest_path}',
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress,
        ):
            for chunk in response.iter_content(chunk_size=const.CHUNK_SIZE):
                if chunk:
                    file.write(chunk)
                    progress.update(len(chunk))
    except requests.RequestException as e:
        raise ConnectionError(f'Failed to download file from {url}. Error: {e}') from e


def verify_file_hash(file_path: str, expected_hash: str) -> bool:
    """Compute the SHA-256 hash of the file and compare it with the expected hash."""
    file_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(const.CHUNK_SIZE), b''):
            file_hash.update(chunk)
    actual_hash = file_hash.hexdigest()
    if actual_hash != expected_hash:
        Logger.warning(f'Expected hash: {expected_hash}, but got: {actual_hash}')
    return actual_hash == expected_hash


if __name__ == '__main__':
    model_names = [model.value for model in const.RetinaFaceWeights]

    # Download each model in the list
    for model_name in model_names:
        model_path = verify_model_weights(model_name)
