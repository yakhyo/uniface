# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


from __future__ import annotations

import numpy as np
import pytest

from uniface.constants import RobustVideoMattingWeights
from uniface.matting import RobustVideoMatting


def test_rvm_initialization():
    """Test RobustVideoMatting initialization with default weights."""
    matting = RobustVideoMatting()
    assert matting is not None
    assert matting.downsample_ratio is None


def test_rvm_with_resnet50_weights_enum():
    """Test the resnet50 enum value maps to the expected model string."""
    assert RobustVideoMattingWeights.RESNET50.value == 'rvm_resnet50'


def test_rvm_invalid_downsample_ratio():
    """Test that out-of-range downsample ratios are rejected."""
    with pytest.raises(ValueError):
        RobustVideoMatting(downsample_ratio=0)
    with pytest.raises(ValueError):
        RobustVideoMatting(downsample_ratio=1.5)


def test_rvm_preprocess():
    """Test preprocessing produces a [0, 1] RGB tensor at the original size."""
    matting = RobustVideoMatting()

    image = np.zeros((128, 160, 3), dtype=np.uint8)
    image[:, :] = (0, 128, 255)  # BGR -> expected RGB (255, 128, 0)
    tensor, orig_h, orig_w = matting.preprocess(image)

    assert tensor.dtype == np.float32
    assert tensor.ndim == 4
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == 3
    assert tensor.shape[2] == 128
    assert tensor.shape[3] == 160
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0
    assert orig_h == 128
    assert orig_w == 160

    # BGR -> RGB channel swap and 0..1 normalization
    assert tensor[0, 0, 0, 0] == pytest.approx(1.0)
    assert tensor[0, 1, 0, 0] == pytest.approx(128 / 255)
    assert tensor[0, 2, 0, 0] == pytest.approx(0.0)


def test_rvm_postprocess():
    """Test postprocessing squeezes and resizes the alpha output."""
    matting = RobustVideoMatting()

    dummy_output = np.random.rand(1, 1, 64, 80).astype(np.float32)
    matte = matting.postprocess(dummy_output, original_size=(80, 64))

    assert matte.shape == (64, 80)
    assert matte.dtype == np.float32

    matte_resized = matting.postprocess(dummy_output, original_size=(160, 128))
    assert matte_resized.shape == (128, 160)


def test_rvm_auto_downsample_ratio():
    """Test automatic downsample ratio keeps the largest side near 512 px."""
    matting = RobustVideoMatting()

    assert matting._resolve_ratio(1920, 1080) == pytest.approx(512 / 1920)
    assert matting._resolve_ratio(512, 512) == 1.0
    assert matting._resolve_ratio(100, 200) == 1.0


def test_rvm_fixed_downsample_ratio():
    """Test a user-provided downsample ratio is used as-is."""
    matting = RobustVideoMatting(downsample_ratio=0.25)

    assert matting._resolve_ratio(1920, 1080) == 0.25
    assert matting._resolve_ratio(100, 200) == 0.25


def test_rvm_reset():
    """Test that reset drops the recurrent state."""
    matting = RobustVideoMatting()

    image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    matting.predict_frame(image)
    assert matting._states is not None

    matting.reset()
    assert matting._states is None
    assert matting._state_key is None


def test_rvm_predict():
    """Test stateless single-image prediction."""
    matting = RobustVideoMatting()

    image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    matte = matting.predict(image)

    assert matte.shape == (128, 128)
    assert matte.dtype == np.float32
    assert matte.min() >= 0.0
    assert matte.max() <= 1.0


def test_rvm_predict_is_stateless():
    """Test that two identical images give identical results (no hidden memory)."""
    matting = RobustVideoMatting()

    image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    matte1 = matting.predict(image)
    matte2 = matting.predict(image)

    assert np.allclose(matte1, matte2)


def test_rvm_predict_frame_sequence():
    """Test stateful frame-by-frame prediction for video."""
    matting = RobustVideoMatting()
    matting.reset()

    size = (96, 128)
    frames = [np.random.randint(0, 255, (*size, 3), dtype=np.uint8) for _ in range(3)]
    mattes = [matting.predict_frame(frame) for frame in frames]

    assert all(matte.shape == size for matte in mattes)
    assert matting._states is not None


def test_rvm_predict_frames():
    """Test batch frame prediction returns a (T, H, W) stack."""
    matting = RobustVideoMatting()
    matting.reset()

    frames = [np.random.randint(0, 255, (96, 128, 3), dtype=np.uint8) for _ in range(3)]
    mattes = matting.predict_frames(frames)

    assert mattes.shape == (3, 96, 128)
    assert mattes.dtype == np.float32


def test_rvm_callable():
    """Test that RobustVideoMatting is callable via __call__."""
    matting = RobustVideoMatting()
    image = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)

    matte = matting(image)

    assert matte.shape == (96, 96)
    assert matte.dtype == np.float32
