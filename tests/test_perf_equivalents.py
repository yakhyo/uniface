# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import itertools
import math

import cv2
import numpy as np
import pytest
from skimage.transform import SimilarityTransform

from uniface.common import decode_landmarks, distance2kps, generate_anchors
from uniface.detection import RetinaFace
from uniface.draw import draw_mesh
from uniface.face_utils import bbox_center_alignment
from uniface.gaze import MobileGaze
from uniface.landmark._tessellation import FACEMESH_TESSELATION_FULL


@pytest.mark.parametrize('image_size', [(640, 640), (480, 640), (333, 500)])
def test_vectorized_generate_anchors_matches_reference(image_size):
    steps = [8, 16, 32]
    min_sizes = [[16, 32], [64, 128], [256, 512]]
    ref = []
    for k, step in enumerate(steps):
        mh, mw = math.ceil(image_size[0] / step), math.ceil(image_size[1] / step)
        for i, j in itertools.product(range(mh), range(mw)):
            for ms in min_sizes[k]:
                ref.extend(
                    [
                        (j + 0.5) * step / image_size[1],
                        (i + 0.5) * step / image_size[0],
                        ms / image_size[1],
                        ms / image_size[0],
                    ]
                )
    expected = np.array(ref, dtype=np.float32).reshape(-1, 4)

    actual = generate_anchors(image_size)
    assert actual.shape == expected.shape
    assert actual.dtype == np.float32
    assert np.allclose(actual, expected, atol=1e-7)


def test_decode_landmarks_and_distance2kps_match_reference():
    rng = np.random.default_rng(7)
    priors = rng.random((100, 4), dtype=np.float32)
    preds = rng.standard_normal((100, 10), dtype=np.float32)

    ref_lms = (
        np.repeat(priors[:, :2][:, None, :], 5, axis=1)
        + preds.reshape(100, 5, 2) * 0.1 * np.repeat(priors[:, 2:][:, None, :], 5, axis=1)
    ).reshape(100, 10)
    assert np.allclose(decode_landmarks(preds, priors), ref_lms, atol=1e-6)

    pts = rng.uniform(0, 640, size=(50, 2)).astype(np.float32)
    dist = rng.uniform(-100, 100, size=(50, 10)).astype(np.float32)
    kps = distance2kps(pts, dist, max_shape=(480, 640))
    assert kps.shape == (50, 10)
    assert np.all((kps[:, 0::2] >= 0) & (kps[:, 0::2] <= 640))
    assert np.all((kps[:, 1::2] >= 0) & (kps[:, 1::2] <= 480))


def test_retinaface_dynamic_size_caches_anchors():
    detector = RetinaFace(dynamic_size=True, providers=['CPUExecutionProvider'])
    image = np.zeros((240, 320, 3), dtype=np.uint8)

    detector.detect(image)
    first_priors = detector._priors
    detector.detect(image)

    assert detector._priors is first_priors


@pytest.mark.parametrize('rotation', [0.0, 27.5, -90.0, 180.0])
def test_bbox_center_alignment_matches_similarity_transform(rotation):
    image = np.random.default_rng(1).integers(0, 256, (300, 400, 3), dtype=np.uint8)
    center = (180.5, 140.25)
    scale = 0.85
    output_size = 192

    rot = rotation * np.pi / 180.0
    t = (
        SimilarityTransform(scale=scale)
        + SimilarityTransform(translation=(-center[0] * scale, -center[1] * scale))
        + SimilarityTransform(rotation=rot)
        + SimilarityTransform(translation=(output_size / 2, output_size / 2))
    )
    expected_m = t.params[0:2]

    _, actual_m = bbox_center_alignment(image, center, output_size, scale, rotation)
    assert np.allclose(actual_m, expected_m, atol=1e-12)


def test_draw_mesh_polylines_matches_line_loop():
    rng = np.random.default_rng(3)
    landmarks = rng.uniform(50, 350, size=(478, 3)).astype(np.float32)
    canvas_poly = np.zeros((400, 400, 3), dtype=np.uint8)
    canvas_ref = np.zeros((400, 400, 3), dtype=np.uint8)

    draw_mesh(canvas_poly, landmarks, mode='full')

    pts = np.rint(landmarks[:, :2]).astype(np.int32)
    for a, b in FACEMESH_TESSELATION_FULL:
        cv2.line(canvas_ref, pts[a], pts[b], (0, 255, 0), 1, cv2.LINE_AA)
    # Also draw irises to match draw_mesh
    for s in (slice(468, 473), slice(473, 478)):
        iris = pts[s]
        r = round(float(np.linalg.norm(iris[1:] - iris[0], axis=1).mean()))
        cv2.circle(canvas_ref, iris[0], max(r, 1), (60, 60, 255), 1, cv2.LINE_AA)
        for x, y in iris:
            cv2.circle(
                canvas_ref,
                (x, y),
                max(round((pts[:468, 1].max() - pts[:468, 1].min()) / 120), 2),
                (60, 60, 255),
                -1,
                cv2.LINE_AA,
            )

    assert np.array_equal(canvas_poly, canvas_ref)


def test_mobile_gaze_preprocess_contiguous_and_estimate():
    gaze = MobileGaze(providers=['CPUExecutionProvider'])
    crop = np.random.default_rng(9).integers(0, 256, (300, 300, 3), dtype=np.uint8)

    tensor = gaze.preprocess(crop)
    assert tensor.shape == (1, 3, 448, 448)
    assert tensor.dtype == np.float32
    assert tensor.flags['C_CONTIGUOUS']

    res = gaze.estimate(crop)
    assert isinstance(res.pitch, float) and isinstance(res.yaw, float)
