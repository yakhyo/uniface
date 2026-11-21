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


def test_elliptical_roi_matches_full_frame_reference_including_borders(image):
    """ROI-cropped elliptical blur reproduces full-frame blur pixel-for-pixel, including edge boxes."""
    import cv2

    bboxes = [[40, 40, 120, 140], [-10, -5, 50, 60], [160, 150, 220, 210]]
    strength, margin = 3.0, 20

    # Reference full-frame calculation
    ref = image.copy()
    h, w = ref.shape[:2]
    for x1, y1, x2, y2 in bboxes:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        ax, ay = (x2 - x1) // 2 + margin, (y2 - y1) // 2 + margin
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)[..., np.newaxis] / 255.0
        ksize = max(3, int((min(ay, ax) * 2 / 7) * strength)) | 1
        blurred = cv2.GaussianBlur(ref, (ksize, ksize), 0)
        ref[:] = (blurred * mask + ref * (1 - mask)).astype(np.uint8)

    actual = BlurFace(method='elliptical', blur_strength=strength, margin=margin).blur_regions(image, bboxes)
    assert np.max(np.abs(actual.astype(np.int16) - ref.astype(np.int16))) <= 1


@pytest.mark.parametrize('bad_bbox', [[100, 100, 100, 100], [120, 120, 80, 80], [-200, -200, -150, -150]])
@pytest.mark.parametrize('method', METHODS)
def test_degenerate_and_offscreen_bboxes_are_skipped_safely(image, method, bad_bbox):
    """Zero-area, inverted, or completely off-screen boxes leave the image unchanged without crashing."""
    result = BlurFace(method=method, margin=-30 if method == 'elliptical' and bad_bbox[0] > 0 else 5).blur_regions(
        image, [bad_bbox]
    )
    assert np.array_equal(result, image)
