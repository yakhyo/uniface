# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


from __future__ import annotations

import numpy as np

from uniface.constants import MODNetWeights
from uniface.matting import MODNet


def test_modnet_initialization():
    """Test MODNet initialization with default weights."""
    matting = MODNet()
    assert matting is not None
    assert matting.input_size == 512


def test_modnet_with_webcam_weights():
    """Test MODNet initialization with webcam variant."""
    matting = MODNet(model_name=MODNetWeights.WEBCAM)
    assert matting is not None
    assert matting.input_size == 512


def test_modnet_custom_input_size():
    """Test MODNet with custom input size."""
    matting = MODNet(input_size=256)
    assert matting.input_size == 256


def test_modnet_preprocess():
    """Test preprocessing produces correct tensor shape and dtype."""
    matting = MODNet()

    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    tensor, orig_h, orig_w = matting.preprocess(image)

    assert tensor.dtype == np.float32
    assert tensor.ndim == 4
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == 3
    assert tensor.shape[2] % 32 == 0
    assert tensor.shape[3] % 32 == 0
    assert orig_h == 480
    assert orig_w == 640


def test_modnet_preprocess_small_image():
    """Test preprocessing with image smaller than input_size."""
    matting = MODNet(input_size=512)

    image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    tensor, orig_h, orig_w = matting.preprocess(image)

    assert tensor.shape[2] % 32 == 0
    assert tensor.shape[3] % 32 == 0
    assert orig_h == 128
    assert orig_w == 128


def test_modnet_preprocess_large_image():
    """Test preprocessing with image larger than input_size."""
    matting = MODNet(input_size=512)

    image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    tensor, orig_h, orig_w = matting.preprocess(image)

    assert tensor.shape[2] % 32 == 0
    assert tensor.shape[3] % 32 == 0
    assert orig_h == 1080
    assert orig_w == 1920


def test_modnet_postprocess():
    """Test postprocessing resizes matte to original dimensions."""
    matting = MODNet()

    dummy_output = np.random.rand(1, 1, 512, 672).astype(np.float32)
    matte = matting.postprocess(dummy_output, original_size=(640, 480))

    assert matte.shape == (480, 640)
    assert matte.dtype == np.float32


def test_modnet_predict():
    """Test end-to-end prediction."""
    matting = MODNet()

    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    matte = matting.predict(image)

    assert matte.shape == (480, 640)
    assert matte.dtype == np.float32
    assert matte.min() >= 0.0
    assert matte.max() <= 1.0


def test_modnet_callable():
    """Test that MODNet is callable via __call__."""
    matting = MODNet()
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    matte = matting(image)

    assert matte.shape == (256, 256)
    assert matte.dtype == np.float32


def test_modnet_different_input_sizes():
    """Test prediction with various image dimensions."""
    matting = MODNet()

    sizes = [(256, 256), (480, 640), (720, 1280), (300, 500)]

    for h, w in sizes:
        image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        matte = matting.predict(image)

        assert matte.shape == (h, w), f'Failed for size {h}x{w}'
        assert matte.dtype == np.float32
