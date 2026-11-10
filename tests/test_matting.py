# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


from __future__ import annotations

import os
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

from uniface.constants import MODNetWeights, PPMattingWeights
from uniface.matting import MODNet, PPMatting
from uniface.model_store import verify_model_weights

PPMATTING_REPO = Path(os.environ.get('PPMATTING_REPO', '~/Workspace/ppmatting')).expanduser()
ASSETS_DIR = Path(__file__).resolve().parent.parent / 'assets' / 'source'


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


@pytest.fixture(scope='module')
def ppmatting():
    """Share one PP-MattingV2 session across the tests that only run inference."""
    return PPMatting()


@pytest.fixture
def standalone_repo():
    """Make the standalone ppmatting repo importable, then undo it so it cannot shadow other modules."""
    repo = str(PPMATTING_REPO)
    modules_before = set(sys.modules)
    sys.path.insert(0, repo)
    try:
        yield PPMATTING_REPO
    finally:
        sys.path.remove(repo)
        for name in set(sys.modules) - modules_before:
            module_file = getattr(sys.modules[name], '__file__', None) or ''
            if module_file.startswith(repo + os.sep):
                del sys.modules[name]


def test_ppmatting_initialization():
    """Test PPMatting initialization with default weights."""
    matting = PPMatting()
    assert matting.max_short == 512
    assert Path(matting.model_path).name == f'{PPMattingWeights.V2.value}.onnx'


def test_ppmatting_v1_weights():
    """Test PPMatting loads and runs the PP-Matting HRNet-W18 variant."""
    matting = PPMatting(model_name=PPMattingWeights.V1)
    assert Path(matting.model_path).name == f'{PPMattingWeights.V1.value}.onnx'

    image = np.random.randint(0, 255, (96, 128, 3), dtype=np.uint8)
    matte = matting.predict(image)

    assert matte.shape == (96, 128)
    assert matte.dtype == np.float32


def test_ppmatting_preprocess_never_upscales(ppmatting):
    """Test an image under max_short is only floored to multiples of 32, never upscaled."""
    image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    tensor, orig_h, orig_w = ppmatting.preprocess(image)

    assert tensor.shape == (1, 3, 192, 288)
    assert tensor.dtype == np.float32
    assert tensor.min() >= -1.0
    assert tensor.max() <= 1.0
    assert (orig_h, orig_w) == (200, 300)


def test_ppmatting_preprocess_500x700(ppmatting):
    """Test a short side of 500 is kept and both sides floored to multiples of 32."""
    image = np.random.randint(0, 255, (500, 700, 3), dtype=np.uint8)
    tensor, orig_h, orig_w = ppmatting.preprocess(image)

    assert tensor.shape == (1, 3, 480, 672)
    assert (orig_h, orig_w) == (500, 700)


def test_ppmatting_preprocess_rounds_long_side(ppmatting):
    """Test the long side is rounded, not truncated, when the short side is shrunk to max_short."""
    # 1535 * 512 / 1024 = 767.5 rounds to 768; truncating would give 767 and then floor to 736.
    image = np.random.randint(0, 255, (1024, 1535, 3), dtype=np.uint8)
    tensor, orig_h, orig_w = ppmatting.preprocess(image)

    assert tensor.shape == (1, 3, 512, 768)
    assert (orig_h, orig_w) == (1024, 1535)


def test_ppmatting_too_small_raises(ppmatting):
    """Test an image with a side under 32 px is rejected."""
    image = np.random.randint(0, 255, (20, 100, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match='32'):
        ppmatting.preprocess(image)


def test_ppmatting_predict_shape_and_range(ppmatting):
    """Test end-to-end prediction returns a matte at the input size in [0, 1]."""
    image = np.random.randint(0, 255, (500, 700, 3), dtype=np.uint8)
    matte = ppmatting.predict(image)

    assert matte.shape == (500, 700)
    assert matte.dtype == np.float32
    assert matte.min() >= 0.0
    assert matte.max() <= 1.0


def test_ppmatting_predict_on_crop_view(ppmatting):
    """Test prediction on a non-contiguous crop view of a larger image."""
    image = np.random.randint(0, 255, (500, 700, 3), dtype=np.uint8)
    view = image[10:490, 20:680]
    assert not view.flags['C_CONTIGUOUS']

    matte = ppmatting.predict(view)

    assert matte.shape == view.shape[:2]
    assert matte.dtype == np.float32


@pytest.mark.skipif(not PPMATTING_REPO.is_dir(), reason=f'standalone ppmatting repo not found at {PPMATTING_REPO}')
def test_ppmatting_matches_standalone_repo(standalone_repo):
    """Test PPMatting reproduces the standalone repo's ONNX inference on a real photo."""
    from model.ppmatting_onnx import PPMattingONNX, preprocess

    image = cv2.imread(str(ASSETS_DIR / 'age_adult.jpg'))
    assert image is not None

    # Both on CPU, so the comparison checks the wrapper rather than execution provider numerics.
    matting = PPMatting(providers=['CPUExecutionProvider'])
    reference = PPMattingONNX(verify_model_weights(PPMattingWeights.V2))

    for sample in (image, np.random.randint(0, 255, (1024, 1535, 3), dtype=np.uint8)):
        expected_tensor, _ = preprocess(sample)
        tensor, _, _ = matting.preprocess(sample)
        np.testing.assert_array_equal(tensor, expected_tensor)

    expected = reference.predict(image)
    matte = matting.predict(image)
    max_abs = float(np.abs(matte - expected).max())
    print(f'PPMatting vs standalone repo on age_adult.jpg: max abs diff {max_abs:.3e}')

    assert matte.shape == expected.shape == image.shape[:2]
    assert max_abs <= 1e-5
