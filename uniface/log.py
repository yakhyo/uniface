import logging

# Create logger for uniface
Logger = logging.getLogger('uniface')
Logger.setLevel(logging.WARNING)  # Only show warnings/errors by default
Logger.addHandler(logging.NullHandler())


def enable_logging(level=logging.INFO):
    """
    Enable verbose logging for uniface.

    Args:
        level: Logging level (logging.DEBUG, logging.INFO, etc.)

    Example:
        >>> from uniface import enable_logging
        >>> enable_logging()  # Show INFO logs
    """
    Logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    Logger.addHandler(handler)
    Logger.setLevel(level)
    Logger.propagate = False
