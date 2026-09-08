# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from uniface.constants import RobustVideoMattingWeights
from uniface.log import Logger
from uniface.model_store import verify_model_weights
from uniface.onnx_utils import create_onnx_session

from .base import BaseMatting

__all__ = ['RobustVideoMatting']

NUM_RECURRENT_STATES = 4
MAX_AUTO_SIDE = 512


class RobustVideoMatting(BaseMatting):
    """Robust Video Matting (RVM) with ONNX Runtime.

    RVM is a recurrent portrait-matting network. Unlike per-frame models such as
    :class:`MODNet`, it carries hidden state across consecutive frames, so the
    alpha matte stays temporally stable on hair and other high-frequency edges
    that make frame-by-frame predictions flicker.

    Two pretrained variants are available:

    - `MOBILENETV3`: fast, recommended for most use cases.
    - `RESNET50`: higher capacity with a modest accuracy gain.

    .. note::
        Weights are distributed under the GPL-3.0 license
        (https://github.com/PeterL1n/RobustVideoMatting); see
        ``docs/license-attribution.md`` before commercial use.

    Raises:
        ValueError: If the model weights are invalid or not found.
        RuntimeError: If the ONNX model fails to load or initialize.

    Reference:
        Lin et al., "Robust High-Resolution Video Matting with Temporal
        Guidance", WACV 2022.
        https://github.com/PeterL1n/RobustVideoMatting

    Args:
        model_name: The enum specifying the RVM variant to load.
            Defaults to `RobustVideoMattingWeights.MOBILENETV3`.
        downsample_ratio: Internal working resolution as a fraction of the
            input size, in ``(0, 1]``. Lower values are faster but coarser.
            If `None` (default), it is picked automatically so the largest
            input side maps to about 512 px, matching the upstream guidance.
        providers: ONNX Runtime execution providers. If `None`, auto-detects
            the best available provider.

    Attributes:
        downsample_ratio (float | None): Downsample ratio used during inference.

    Example:
        >>> from uniface.matting import RobustVideoMatting
        >>>
        >>> # Single independent image (no temporal memory is used)
        >>> matting = RobustVideoMatting()
        >>> matte = matting.predict(image)  # (H, W) float32 in [0, 1]
        >>>
        >>> # Video: carry memory across frames for a stable matte
        >>> matting.reset()
        >>> for frame in frames:
        ...     matte = matting.predict_frame(frame)
    """

    def __init__(
        self,
        *,
        model_name: RobustVideoMattingWeights = RobustVideoMattingWeights.MOBILENETV3,
        downsample_ratio: float | None = None,
        providers: list[str] | None = None,
    ) -> None:
        if downsample_ratio is not None and not 0.0 < downsample_ratio <= 1.0:
            raise ValueError('downsample_ratio must be in (0, 1] or None for auto.')

        Logger.info(f'Initializing RobustVideoMatting with model={model_name}')

        self.downsample_ratio = downsample_ratio
        self.providers = providers

        self.model_path = verify_model_weights(model_name)
        self._initialize_model()

        # Recurrent state carried between predict_frame calls. None means "no
        # memory yet", which the model initializes internally from broadcastable
        # zero states on the next call.
        self._states: list[np.ndarray] | None = None
        self._state_key: tuple[int, int, float] | None = None

    def _initialize_model(self) -> None:
        """Initialize the ONNX model from the stored model path.

        The RVM ONNX graph has one image input, one downsample-ratio input,
        four recurrent-state inputs, and the matching outputs. Nodes are
        discovered from shapes rather than hard-coded names so the class keeps
        working if the graph is re-exported with different tensor names.

        Raises:
            RuntimeError: If the model fails to load or initialize.
        """
        try:
            self.session = create_onnx_session(self.model_path, providers=self.providers)

            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()

            self.src_input = _channel_input(inputs, 3)
            self.ratio_input = next(i for i in inputs if len(i.shape) == 1)
            self.state_inputs = [i for i in inputs if i is not self.src_input and i is not self.ratio_input]

            self.fgr_output = _channel_output(outputs, 3)
            self.pha_output = _channel_output(outputs, 1)
            self.state_outputs = [o for o in outputs if o is not self.fgr_output and o is not self.pha_output]

            if len(self.state_inputs) != NUM_RECURRENT_STATES or len(self.state_outputs) != NUM_RECURRENT_STATES:
                raise RuntimeError(
                    f'Expected {NUM_RECURRENT_STATES} recurrent states, '
                    f'got {len(self.state_inputs)} inputs / {len(self.state_outputs)} outputs.'
                )

            Logger.info('RobustVideoMatting initialized')

        except Exception as e:
            Logger.error(f"Failed to load RobustVideoMatting model from '{self.model_path}'", exc_info=True)
            raise RuntimeError(f'Failed to initialize RobustVideoMatting model: {e}') from e

    def reset(self) -> None:
        """Drop the recurrent memory.

        Call this at the start of a new video or after a hard scene cut so
        predictions do not inherit state from the previous sequence.
        """
        self._states = None
        self._state_key = None
        Logger.debug('RobustVideoMatting recurrent state reset')

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Preprocess a BGR image for RVM inference.

        RVM expects RGB input normalized to ``[0, 1]`` at the original spatial
        resolution; the model downsamples internally per `downsample_ratio`.
        No letterboxing or padding is applied.

        Args:
            image: Input image in BGR format with shape `(H, W, 3)`.

        Returns:
            A tuple of `(tensor, orig_h, orig_w)` where *tensor* has shape
            `(1, 3, H, W)` in float32.
        """
        orig_h, orig_w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        return np.expand_dims(x, axis=0), orig_h, orig_w

    def postprocess(self, outputs: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
        """Postprocess raw model output into an alpha matte.

        RVM emits the alpha at the source resolution already, so this only
        squeezes the batch/channel axes and, as a safeguard, resizes to the
        original size.

        Args:
            outputs: Raw ONNX alpha output with shape `(1, 1, H, W)`.
            original_size: Target size as `(width, height)`.

        Returns:
            Alpha matte with shape `(H, W)`, float32 in `[0, 1]`.
        """
        matte = outputs[0, 0]
        matte = cv2.resize(matte, original_size, interpolation=cv2.INTER_AREA)
        return matte

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run portrait matting on a single image without temporal memory.

        Predictions for unrelated images should stay independent, so this call
        always starts from a fresh recurrent state and does not persist it.
        For video use :meth:`reset` plus :meth:`predict_frame`, or
        :meth:`predict_frames`.

        Args:
            image: Input image in BGR format with shape `(H, W, 3)`.

        Returns:
            Alpha matte with shape `(H, W)`, float32 in `[0, 1]`.
        """
        tensor, orig_h, orig_w = self.preprocess(image)
        ratio = self._resolve_ratio(orig_h, orig_w)
        _, pha, _ = self._run(tensor, ratio, states=None)
        return self.postprocess(pha, (orig_w, orig_h))

    def predict_frame(self, image: np.ndarray) -> np.ndarray:
        """Run matting on the next frame of a video, reusing temporal memory.

        Call :meth:`reset` before the first frame (or after a scene cut), then
        feed frames in order. Memory is kept across calls so the matte stays
        stable from frame to frame.

        Args:
            image: Input frame in BGR format with shape `(H, W, 3)`.

        Returns:
            Alpha matte with shape `(H, W)`, float32 in `[0, 1]`.
        """
        tensor, orig_h, orig_w = self.preprocess(image)
        ratio = self._resolve_ratio(orig_h, orig_w)
        key = (orig_h, orig_w, ratio)
        if self._states is not None and key != self._state_key:
            Logger.debug('RobustVideoMatting frame geometry changed; resetting recurrent state')
            self.reset()
        _, pha, states = self._run(tensor, ratio, states=self._states)
        self._states = states
        self._state_key = key
        return self.postprocess(pha, (orig_w, orig_h))

    def predict_frames(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        """Run matting on a sequence of frames with shared temporal memory.

        Convenience wrapper around :meth:`predict_frame`. Geometry changes
        between frames reset the memory automatically. Recurrent state is
        retained across calls for chunked processing; call :meth:`reset` before
        starting an unrelated sequence.

        Args:
            frames: Iterable of BGR frames, each with shape `(H, W, 3)`.

        Returns:
            Alpha mattes stacked along a new leading axis with shape `(T, H, W)`,
            float32 in `[0, 1]`.
        """
        mattes = [self.predict_frame(frame) for frame in frames]
        return np.stack(mattes, axis=0)

    def _resolve_ratio(self, h: int, w: int) -> float:
        """Return the downsample ratio for a frame of the given size."""
        if self.downsample_ratio is not None:
            return self.downsample_ratio
        # Upstream auto rule: map the largest side down to ~512 px.
        return min(MAX_AUTO_SIDE / max(h, w), 1.0)

    def _run(
        self,
        src: np.ndarray,
        ratio: float,
        states: list[np.ndarray] | None,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray] | None]:
        """Run one forward pass, optionally recycling recurrent states.

        Args:
            src: Preprocessed source tensor with shape `(1, 3, H, W)`.
            ratio: Downsample ratio for this pass.
            states: Previous recurrent states, or `None` for a fresh start.

        Returns:
            Tuple of `(fgr, pha, new_states)` where *new_states* are the
            recurrent outputs of this pass and can be fed back on the next frame.
        """
        if states is None:
            # The exported graph broadcasts 1x1x1x1 zeros to the internal state
            # geometry, so a shape-generic zero init works for any resolution.
            zeros = [np.zeros((1, 1, 1, 1), dtype=np.float32) for _ in range(NUM_RECURRENT_STATES)]
            states = zeros

        feeds = dict(zip([i.name for i in self.state_inputs], states, strict=True))
        feeds[self.src_input.name] = src
        feeds[self.ratio_input.name] = np.asarray([ratio], dtype=np.float32)

        outputs = self.session.run(
            [self.fgr_output.name, self.pha_output.name] + [o.name for o in self.state_outputs],
            feeds,
        )
        fgr, pha = outputs[0], outputs[1]
        new_states = list(outputs[2:])
        return fgr, pha, new_states


def _channel_input(inputs, channels: int):
    """Return the graph input whose spatial channels match `channels`."""
    for i in inputs:
        shape = i.shape
        if len(shape) == 4 and shape[1] == channels:
            return i
    raise RuntimeError(f'No input tensor with {channels} channels found in model.')


def _channel_output(outputs, channels: int):
    """Return the graph output whose spatial channels match `channels`."""
    for o in outputs:
        shape = o.shape
        if len(shape) == 4 and shape[1] == channels:
            return o
    raise RuntimeError(f'No output tensor with {channels} channels found in model.')
