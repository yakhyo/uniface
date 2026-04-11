# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import cv2
import numpy as np

from uniface.constants import MODNetWeights
from uniface.log import Logger
from uniface.model_store import verify_model_weights
from uniface.onnx_utils import create_onnx_session

from .base import BaseMatting

__all__ = ['MODNet']

STRIDE = 32


class MODNet(BaseMatting):
    """MODNet: Real-Time Trimap-Free Portrait Matting with ONNX Runtime.

    MODNet produces a soft alpha matte from a full image without requiring
    a trimap. It uses a MobileNetV2 backbone with low-resolution, high-resolution,
    and fusion branches to generate accurate mattes at real-time speed.

    Two pretrained variants are available:

    - ``PHOTOGRAPHIC``: optimized for high-quality portrait photos.
    - ``WEBCAM``: optimized for real-time webcam feeds.

    Reference:
        Ke et al., "MODNet: Real-Time Trimap-Free Portrait Matting via
        Objective Decomposition", AAAI 2022.
        https://github.com/yakhyo/modnet

    Args:
        model_name: The enum specifying the MODNet variant to load.
            Defaults to ``MODNetWeights.PHOTOGRAPHIC``.
        input_size: Target size for the shorter side during preprocessing.
            The image is resized so its shorter side equals this value
            (aspect ratio preserved), then both dimensions are floored to
            multiples of 32. Defaults to 512.
        providers: ONNX Runtime execution providers. If ``None``, auto-detects
            the best available provider.

    Attributes:
        input_size (int): Target shorter-side size for preprocessing.

    Example:
        >>> from uniface.matting import MODNet
        >>>
        >>> matting = MODNet()
        >>> matte = matting.predict(image)  # (H, W) float32 in [0, 1]
        >>>
        >>> # Composite onto green background
        >>> import numpy as np
        >>> bg = np.full_like(image, (0, 177, 64), dtype=np.uint8)
        >>> alpha = matte[..., np.newaxis]
        >>> result = (image * alpha + bg * (1 - alpha)).astype(np.uint8)
    """

    def __init__(
        self,
        model_name: MODNetWeights = MODNetWeights.PHOTOGRAPHIC,
        input_size: int = 512,
        providers: list[str] | None = None,
    ) -> None:
        Logger.info(f'Initializing MODNet with model={model_name}, input_size={input_size}')

        self.input_size = input_size
        self.providers = providers

        self.model_path = verify_model_weights(model_name)
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the ONNX model from the stored model path.

        Raises:
            RuntimeError: If the model fails to load or initialize.
        """
        try:
            self.session = create_onnx_session(self.model_path, providers=self.providers)

            input_cfg = self.session.get_inputs()[0]
            self.input_name = input_cfg.name

            outputs = self.session.get_outputs()
            self.output_names = [output.name for output in outputs]

            Logger.info(f'MODNet initialized with input_size={self.input_size}')

        except Exception as e:
            Logger.error(f"Failed to load MODNet model from '{self.model_path}'", exc_info=True)
            raise RuntimeError(f'Failed to initialize MODNet model: {e}') from e

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Preprocess a BGR image for MODNet inference.

        The image is converted to RGB, resized so its shorter side matches
        ``input_size`` (aspect ratio preserved), floored to multiples of 32,
        and normalized to ``[-1, 1]``.

        Args:
            image: Input image in BGR format with shape ``(H, W, 3)``.

        Returns:
            A tuple of ``(tensor, orig_h, orig_w)`` where *tensor* has shape
            ``(1, 3, H', W')`` in float32.
        """
        orig_h, orig_w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if max(orig_h, orig_w) < self.input_size or min(orig_h, orig_w) > self.input_size:
            if orig_w >= orig_h:
                new_h = self.input_size
                new_w = int(orig_w / orig_h * self.input_size)
            else:
                new_w = self.input_size
                new_h = int(orig_h / orig_w * self.input_size)
        else:
            new_h, new_w = orig_h, orig_w

        new_h = new_h - (new_h % STRIDE)
        new_w = new_w - (new_w % STRIDE)
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        x = rgb.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = np.transpose(x, (2, 0, 1))

        return np.expand_dims(x, axis=0), orig_h, orig_w

    def postprocess(self, outputs: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
        """Postprocess raw model output into an alpha matte.

        Args:
            outputs: Raw ONNX output with shape ``(1, 1, H', W')``.
            original_size: Target size as ``(width, height)``.

        Returns:
            Alpha matte with shape ``(H, W)``, float32 in ``[0, 1]``.
        """
        matte = outputs[0, 0]
        matte = cv2.resize(matte, original_size, interpolation=cv2.INTER_AREA)
        return matte

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run portrait matting on a BGR image.

        Args:
            image: Input image in BGR format with shape ``(H, W, 3)``.

        Returns:
            Alpha matte with shape ``(H, W)``, float32 in ``[0, 1]``.
        """
        tensor, orig_h, orig_w = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        return self.postprocess(outputs[0], (orig_w, orig_h))
