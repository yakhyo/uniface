# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""ONNX Runtime utilities for UniFace.

This module provides helper functions for creating and managing ONNX Runtime
inference sessions with automatic hardware acceleration detection.
"""

from __future__ import annotations

import onnxruntime as ort

from uniface.log import Logger

__all__ = ['create_onnx_session', 'get_available_providers']


def get_available_providers() -> list[str]:
    """Get list of available ONNX Runtime execution providers.

    Automatically detects and prioritizes hardware acceleration:
    - CoreML on Apple Silicon (M1/M2/M3/M4)
    - CUDA on NVIDIA GPUs
    - CPU as fallback (always available)

    Returns:
        Ordered list of execution providers to use.

    Example:
        >>> providers = get_available_providers()
        >>> # On M4 Mac: ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        >>> # On Linux with CUDA: ['CUDAExecutionProvider', 'CPUExecutionProvider']
    """
    available = ort.get_available_providers()
    providers = []

    # Priority order: CoreML > CUDA > CPU
    if 'CoreMLExecutionProvider' in available:
        providers.append('CoreMLExecutionProvider')
        Logger.info('CoreML acceleration enabled (Apple Silicon)')

    if 'CUDAExecutionProvider' in available:
        providers.append('CUDAExecutionProvider')
        Logger.info('CUDA acceleration enabled (NVIDIA GPU)')

    # CPU is always available as fallback
    providers.append('CPUExecutionProvider')

    if len(providers) == 1:
        Logger.info('Using CPU execution (no hardware acceleration detected)')

    return providers


def create_onnx_session(
    model_path: str,
    providers: list[str] | None = None,
) -> ort.InferenceSession:
    """Create an ONNX Runtime inference session with optimal provider selection.

    Args:
        model_path: Path to the ONNX model file.
        providers: List of execution providers to use. If None, automatically
            detects best available providers.

    Returns:
        Configured ONNX Runtime session.

    Raises:
        RuntimeError: If session creation fails.

    Example:
        >>> session = create_onnx_session('model.onnx')
        >>> # Automatically uses best available providers

        >>> session = create_onnx_session('model.onnx', providers=['CPUExecutionProvider'])
        >>> # Force CPU-only execution
    """
    if providers is None:
        providers = get_available_providers()

    # Suppress ONNX Runtime warnings (e.g., CoreML partition warnings)
    # Log levels: 0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3  # Only show ERROR and FATAL

    try:
        session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        active_provider = session.get_providers()[0]
        Logger.debug(f'Session created with provider: {active_provider}')

        # Show user-friendly message about which provider is being used
        provider_names = {
            'CoreMLExecutionProvider': 'CoreML (Apple Silicon)',
            'CUDAExecutionProvider': 'CUDA (NVIDIA GPU)',
            'CPUExecutionProvider': 'CPU',
        }
        provider_display = provider_names.get(active_provider, active_provider)
        Logger.info(f'✓ Model loaded ({provider_display})')

        return session
    except Exception as e:
        Logger.error(f'Failed to create ONNX session: {e}', exc_info=True)
        raise RuntimeError(f'Failed to initialize ONNX Runtime session: {e}') from e
