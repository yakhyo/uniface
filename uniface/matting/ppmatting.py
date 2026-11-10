# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import cv2
import numpy as np

from uniface.constants import PPMattingWeights
from uniface.log import Logger
from uniface.model_store import verify_model_weights
from uniface.onnx_utils import create_onnx_session

from .base import BaseMatting

__all__ = ['PPMatting']

STRIDE = 32


class PPMatting(BaseMatting):
    """PP-Matting: Trimap-Free Portrait Matting with ONNX Runtime.

    PP-Matting produces a soft alpha matte from a full image without requiring
    a trimap. A semantic branch classifies each pixel as foreground, background
    or transition, and a detail branch refines the alpha in the transition
    region, so hair and soft edges keep their partial opacity.

    Two pretrained variants are available:

    - `V2`: PP-MattingV2 with an STDC1 backbone, faster.
    - `V1`: PP-Matting with an HRNet-W18 backbone, more accurate but slower.

    Raises:
        ValueError: If the model weights are invalid or not found.
        RuntimeError: If the ONNX model fails to load or initialize.

    Reference:
        Chen et al., "PP-Matting: High-Accuracy Natural Image Matting",
        arXiv:2204.09433, 2022.
        https://github.com/yakhyo/ppmatting

    Args:
        model_name: The enum specifying the PP-Matting variant to load.
            Defaults to `PPMattingWeights.V2`.
        max_short: Upper bound on the shorter side during preprocessing.
            Larger images are shrunk to it (aspect ratio preserved), smaller
            images are not upscaled, and both sides are then floored to
            multiples of 32. Defaults to 512.
        providers: ONNX Runtime execution providers. If `None`, auto-detects
            the best available provider.

    Attributes:
        max_short (int): Upper bound on the shorter side for preprocessing.

    Example:
        >>> from uniface.matting import PPMatting
        >>>
        >>> matting = PPMatting()
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
        *,
        model_name: PPMattingWeights = PPMattingWeights.V2,
        max_short: int = 512,
        providers: list[str] | None = None,
    ) -> None:
        Logger.info(f'Initializing PPMatting with model={model_name}, max_short={max_short}')

        self.max_short = max_short
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

            Logger.info(f'PPMatting initialized with max_short={self.max_short}')

        except Exception as e:
            Logger.error(f"Failed to load PPMatting model from '{self.model_path}'", exc_info=True)
            raise RuntimeError(f'Failed to initialize PPMatting model: {e}') from e

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Preprocess a BGR image for PP-Matting inference.

        Converts to RGB, caps the shorter side at `max_short` without
        upscaling, floors both sides to multiples of 32 and normalizes
        the pixels to `[-1, 1]`.

        Args:
            image: Input image in BGR format with shape `(H, W, 3)`.

        Returns:
            A tuple of `(tensor, orig_h, orig_w)` where *tensor* has shape
            `(1, 3, H', W')` in float32.

        Raises:
            ValueError: If either side of the image is shorter than 32 pixels.
        """
        orig_h, orig_w = image.shape[:2]
        if orig_h < STRIDE or orig_w < STRIDE:
            raise ValueError(f'PPMatting needs an image of at least {STRIDE}x{STRIDE} pixels, got {orig_h}x{orig_w}')

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Shrink the shorter side to max_short, rounding the other side as PaddleSeg does.
        short = min(orig_h, orig_w)
        if short > self.max_short:
            scale = float(self.max_short) / float(short)
            new_w, new_h = round(orig_w * scale), round(orig_h * scale)
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Floor both sides to multiples of the network stride.
        h, w = rgb.shape[:2]
        new_h, new_w = h - h % STRIDE, w - w % STRIDE
        if (new_h, new_w) != (h, w):
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        x = rgb.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = np.transpose(x, (2, 0, 1))

        return np.ascontiguousarray(x[np.newaxis]), orig_h, orig_w

    def postprocess(self, outputs: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
        """Postprocess raw model output into an alpha matte.

        Args:
            outputs: Raw ONNX output with shape `(1, 1, H', W')`.
            original_size: Target size as `(width, height)`.

        Returns:
            Alpha matte with shape `(H, W)`, float32 in `[0, 1]`.
        """
        matte = outputs[0, 0]
        matte = cv2.resize(matte, original_size, interpolation=cv2.INTER_LINEAR)
        return matte

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run portrait matting on a BGR image.

        Args:
            image: Input image in BGR format with shape `(H, W, 3)`.

        Returns:
            Alpha matte with shape `(H, W)`, float32 in `[0, 1]`.
        """
        tensor, orig_h, orig_w = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        return self.postprocess(outputs[0], (orig_w, orig_h))
