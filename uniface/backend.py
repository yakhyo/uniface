# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo
#
# Backend Selection System for UniFace

"""
Backend selection system for UniFace.

This module provides automatic backend selection based on the platform:
- Apple Silicon (M1/M2/M3/M4): Use MLX for best performance
- Other platforms: Use ONNX Runtime

Users can also manually override the backend selection.
"""

from enum import Enum
from typing import Optional

from uniface.log import Logger

__all__ = [
    'Backend',
    'get_backend',
    'set_backend',
    'get_available_backends',
    'is_backend_available',
]


class Backend(str, Enum):
    """Available inference backends."""

    MLX = 'mlx'
    ONNX = 'onnx'
    AUTO = 'auto'


# Global backend setting
_current_backend: Optional[Backend] = None


def _check_mlx_available() -> bool:
    """Check if MLX backend is available."""
    try:
        from uniface.mlx_utils import is_mlx_available

        return is_mlx_available()
    except ImportError:
        return False


def _check_onnx_available() -> bool:
    """Check if ONNX Runtime backend is available."""
    try:
        import onnxruntime  # noqa: F401

        return True
    except ImportError:
        return False


def get_available_backends() -> list:
    """
    Get list of available backends on this system.

    Returns:
        List of available Backend enum values.
    """
    available = []

    if _check_mlx_available():
        available.append(Backend.MLX)

    if _check_onnx_available():
        available.append(Backend.ONNX)

    return available


def is_backend_available(backend: Backend) -> bool:
    """
    Check if a specific backend is available.

    Args:
        backend: Backend to check.

    Returns:
        True if the backend is available, False otherwise.
    """
    if backend == Backend.AUTO:
        return len(get_available_backends()) > 0
    elif backend == Backend.MLX:
        return _check_mlx_available()
    elif backend == Backend.ONNX:
        return _check_onnx_available()
    return False


def get_backend() -> Backend:
    """
    Get the current backend.

    If no backend is explicitly set, auto-selects based on platform:
    - Apple Silicon with MLX installed: MLX
    - Otherwise: ONNX Runtime

    Returns:
        The current or auto-selected Backend.

    Raises:
        RuntimeError: If no backend is available.
    """
    global _current_backend

    if _current_backend is not None and _current_backend != Backend.AUTO:
        return _current_backend

    # Auto-select best backend
    available = get_available_backends()

    if not available:
        raise RuntimeError(
            'No inference backend available. '
            'Install either MLX (pip install mlx) on Apple Silicon, '
            'or ONNX Runtime (pip install onnxruntime).'
        )

    # Prefer MLX on Apple Silicon, otherwise ONNX
    if Backend.MLX in available:
        selected = Backend.MLX
        Logger.info('Auto-selected MLX backend (Apple Silicon detected)')
    else:
        selected = Backend.ONNX
        Logger.info('Auto-selected ONNX Runtime backend')

    return selected


def set_backend(backend: Backend) -> None:
    """
    Manually set the inference backend.

    Args:
        backend: The backend to use (MLX, ONNX, or AUTO).

    Raises:
        ValueError: If the specified backend is not available.
    """
    global _current_backend

    if backend != Backend.AUTO and not is_backend_available(backend):
        available = get_available_backends()
        raise ValueError(
            f"Backend '{backend.value}' is not available. Available backends: {[b.value for b in available]}"
        )

    _current_backend = backend
    Logger.info(f'Backend set to: {backend.value}')


def reset_backend() -> None:
    """Reset backend to auto-selection mode."""
    global _current_backend
    _current_backend = None
    Logger.debug('Backend reset to auto-selection')


def use_mlx() -> bool:
    """
    Check if MLX should be used for inference.

    Convenience function for conditional imports.

    Returns:
        True if MLX is the current backend, False otherwise.
    """
    try:
        return get_backend() == Backend.MLX
    except RuntimeError:
        return False


def use_onnx() -> bool:
    """
    Check if ONNX Runtime should be used for inference.

    Convenience function for conditional imports.

    Returns:
        True if ONNX is the current backend, False otherwise.
    """
    try:
        return get_backend() == Backend.ONNX
    except RuntimeError:
        return False
