# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from typing_extensions import deprecated

from uniface.constants import MODNetWeights

from .base import BaseMatting
from .modnet import MODNet

__all__ = ['BaseMatting', 'MODNet', 'create_matting_model']


@deprecated(
    'create_matting_model() is deprecated and will be removed in uniface 4.0. '
    'Instantiate MODNet directly, e.g. '
    '`from uniface.matting import MODNet; MODNet(model_name=...)`.'
)
def create_matting_model(
    model_name: str | MODNetWeights = MODNetWeights.PHOTOGRAPHIC,
    **kwargs,
) -> BaseMatting:
    """Factory function to create a portrait matting model.

    .. deprecated:: 3.7.0
        Use ``MODNet`` directly. This factory will be removed in uniface 4.0.

    Args:
        model_name: Model to create. Options: ``MODNetWeights.PHOTOGRAPHIC``
            (high-quality photos), ``MODNetWeights.WEBCAM`` (real-time webcam).
            Also accepts string values like ``"modnet_photographic"`` or
            ``"modnet_webcam"``.
        **kwargs: Additional arguments passed to the model constructor
            (e.g. ``input_size``, ``providers``).

    Returns:
        An instance of the requested matting model.

    Raises:
        ValueError: If the model_name is not recognized.

    Example:
        >>> from uniface.matting import MODNet
        >>> matting = MODNet()
        >>> matte = matting.predict(image)
    """
    if isinstance(model_name, MODNetWeights):
        return MODNet(model_name=model_name, **kwargs)

    if isinstance(model_name, str):
        try:
            weights = MODNetWeights(model_name)
            return MODNet(model_name=weights, **kwargs)
        except ValueError:
            pass

        valid_models = [m.value for m in MODNetWeights]
        raise ValueError(f"Unknown matting model: '{model_name}'. Valid options are: {', '.join(valid_models)}")

    valid_models = [m.value for m in MODNetWeights]
    raise ValueError(f"Unknown matting model: '{model_name}'. Valid options are: {', '.join(valid_models)}")
