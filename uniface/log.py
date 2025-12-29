# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Logging utilities for UniFace.

This module provides a centralized logger for the UniFace library,
allowing users to enable verbose logging when debugging or developing.
"""

from __future__ import annotations

import logging

__all__ = ['Logger', 'enable_logging']

# Create logger for uniface
Logger = logging.getLogger('uniface')
Logger.setLevel(logging.WARNING)  # Only show warnings/errors by default
Logger.addHandler(logging.NullHandler())


def enable_logging(level: int = logging.INFO) -> None:
    """Enable verbose logging for uniface.

    Configures the logger to output messages to stdout with timestamps.
    Call this function to see informational messages during model loading
    and inference.

    Args:
        level: Logging level. Defaults to logging.INFO.
            Common values: logging.DEBUG, logging.INFO, logging.WARNING.

    Example:
        >>> from uniface import enable_logging
        >>> import logging
        >>> enable_logging()  # Show INFO logs
        >>> enable_logging(level=logging.DEBUG)  # Show DEBUG logs
    """
    Logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    Logger.addHandler(handler)
    Logger.setLevel(level)
    Logger.propagate = False
