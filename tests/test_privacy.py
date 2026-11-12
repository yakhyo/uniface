# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import numpy as np
import pytest

from uniface.privacy import BlurFace

METHODS = ['gaussian', 'median', 'pixelate', 'blackout', 'elliptical']
BBOX = [40, 40, 120, 140]


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(200, 200, 3), dtype=np.uint8)


@pytest.mark.parametrize('method', METHODS)
def test_inplace_modifies_the_input_image(image, method):
    """With inplace=True the caller's array is blurred and returned."""
    original = image.copy()

    result = BlurFace(method=method).blur_regions(image, [BBOX], inplace=True)

    assert result is image
    assert not np.array_equal(image, original)


@pytest.mark.parametrize('method', METHODS)
def test_default_leaves_the_input_image_untouched(image, method):
    """Without inplace the caller's array is kept and a blurred copy is returned."""
    original = image.copy()

    result = BlurFace(method=method).blur_regions(image, [BBOX])

    assert result is not image
    assert np.array_equal(image, original)
    assert not np.array_equal(result, original)
