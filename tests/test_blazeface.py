# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from pathlib import Path
import threading

import cv2
import numpy as np
import pytest

from uniface.constants import BlazeFaceWeights
from uniface.detection import BlazeFace
from uniface.detection.blazeface import _weighted_nms

TEST_IMAGE = Path(__file__).resolve().parent.parent / 'assets' / 'source' / 'verify_einstein_1947.jpg'


@pytest.fixture(scope='module')
def blazeface_model():
    return BlazeFace(model_name=BlazeFaceWeights.DEFAULT)


@pytest.fixture(scope='module')
def face_image():
    image = cv2.imread(str(TEST_IMAGE))
    assert image is not None, f'Missing test asset: {TEST_IMAGE}'
    return image


def test_model_initialization(blazeface_model):
    assert blazeface_model is not None
    assert blazeface_model.input_size == 128
    assert blazeface_model.anchors.shape == (896, 2)


def test_does_not_support_alignment(blazeface_model):
    """The contract that keeps BlazeFace out of recognition pipelines."""
    assert blazeface_model.supports_alignment is False
    assert blazeface_model.supports_landmarks is True
    assert blazeface_model.get_info()['supports_alignment'] is False


def test_detect_returns_faces(blazeface_model, face_image):
    faces = blazeface_model.detect(face_image)

    assert isinstance(faces, list)
    assert len(faces) >= 1

    for face in faces:
        assert 0.0 <= face.confidence <= 1.0


def test_emits_six_keypoints(blazeface_model, face_image):
    """MediaPipe's layout, not the 5-point alignment template."""
    face = blazeface_model.detect(face_image)[0]

    assert face.landmarks.shape == (6, 2)
    assert face.landmarks.dtype == np.float32


def test_first_two_keypoints_are_the_eyes(blazeface_model, face_image):
    """Row 0 is the viewer-left eye, row 1 the viewer-right — what FaceMesh reads."""
    face = blazeface_model.detect(face_image)[0]

    assert face.landmarks[0][0] < face.landmarks[1][0]


def test_bbox_is_well_formed(blazeface_model, face_image):
    """Boxes are ordered and non-degenerate, but not clipped to the frame."""
    for face in blazeface_model.detect(face_image):
        x1, y1, x2, y2 = face.bbox
        assert x1 < x2 and y1 < y2


def test_bbox_stays_square_when_the_face_runs_off_frame(blazeface_model, face_image):
    """The box is left unclipped so FaceMesh's square-ROI rule still holds.

    MediaPipe's `detection_to_roi` expands a *square* box by 1.5x. Clipping to the
    frame breaks the squareness and shifts the centre, which drags the mesh off the
    mouth on webcam close-ups — the exact case where the box overflows.
    """
    # Crop just around the detected face, so the decoded box overflows the frame. Derived
    # from the detection rather than hardcoded, so replacing the test image cannot quietly
    # turn this into a crop of the background.
    detected = blazeface_model.detect(face_image)
    assert detected, 'expected a detection on the full frame'
    x1, y1, x2, y2 = (int(v) for v in detected[0].bbox)
    pad_x, pad_y = round((x2 - x1) * 0.05), round((y2 - y1) * 0.05)
    frame_height, frame_width = face_image.shape[:2]
    frame = cv2.resize(
        face_image[
            max(0, y1 - pad_y) : min(frame_height, y2 + pad_y), max(0, x1 - pad_x) : min(frame_width, x2 + pad_x)
        ],
        (640, 480),
    )
    height, width = frame.shape[:2]

    faces = blazeface_model.detect(frame)
    assert faces, 'expected a detection on the close-up crop'

    x1, y1, x2, y2 = faces[0].bbox
    assert x1 < 0 or y1 < 0 or x2 > width or y2 > height, 'crop is not tight enough to exercise overflow'
    assert (y2 - y1) / (x2 - x1) == pytest.approx(1.0, abs=0.02)


def test_no_faces_in_blank_images(blazeface_model):
    """Also covers letterboxing in both orientations."""
    for shape in ((480, 640, 3), (640, 480, 3)):
        assert blazeface_model.detect(np.zeros(shape, dtype=np.uint8)) == []


@pytest.mark.parametrize(
    ('label', 'bad_image'),
    [
        ('float [0, 1]', np.random.rand(480, 640, 3).astype(np.float32)),
        ('grayscale', np.zeros((480, 640), dtype=np.uint8)),
        ('empty', np.zeros((0, 0, 3), dtype=np.uint8)),
    ],
)
def test_detect_rejects_unusable_images(blazeface_model, label, bad_image):
    """This detector warps its own input, bypassing the shared resize helpers.

    A float image used to return zero faces silently and a grayscale one raised
    an opaque IndexError, both unlike every other detector.
    """
    with pytest.raises(ValueError):
        blazeface_model.detect(bad_image)


def test_no_overflow_warning_on_extreme_logits(blazeface_model, face_image):
    """Anchor logits are clipped before the sigmoid; numpy must not warn."""
    with np.errstate(all='raise'):
        blazeface_model.detect(face_image)


def test_weighted_nms_blends_overlapping_boxes():
    """MediaPipe averages overlapping candidates rather than discarding them."""
    columns = 5 + 2 * 6
    rows = np.zeros((2, columns))
    rows[0, :5] = [0.9, 0.50, 0.50, 0.20, 0.20]
    rows[1, :5] = [0.3, 0.52, 0.52, 0.20, 0.20]

    merged = _weighted_nms(rows, iou_threshold=0.3)

    assert len(merged) == 1
    assert merged[0, 0] == pytest.approx(0.9)
    # Pulled toward the weaker box rather than ignoring it.
    assert 0.50 < merged[0, 1] < 0.52


def test_weighted_nms_keeps_disjoint_boxes():
    columns = 5 + 2 * 6
    rows = np.zeros((2, columns))
    rows[0, :5] = [0.9, 0.20, 0.20, 0.10, 0.10]
    rows[1, :5] = [0.8, 0.80, 0.80, 0.10, 0.10]

    assert len(_weighted_nms(rows, iou_threshold=0.3)) == 2


def _nms_within(rows: np.ndarray, iou_threshold: float, seconds: float = 5.0) -> np.ndarray:
    """Run `_weighted_nms` on a worker thread so a non-terminating loop fails instead of hanging."""
    result: list[np.ndarray] = []
    worker = threading.Thread(target=lambda: result.append(_weighted_nms(rows, iou_threshold)), daemon=True)
    worker.start()
    worker.join(seconds)

    assert not worker.is_alive(), f'_weighted_nms did not terminate within {seconds}s'
    return result[0]


def test_weighted_nms_terminates_when_threshold_excludes_self_overlap():
    """iou_threshold=1.0 fails the strict `>` even for the winner's self-IoU of 1.0."""
    columns = 5 + 2 * 6
    rows = np.zeros((2, columns))
    rows[0, :5] = [0.9, 0.50, 0.50, 0.20, 0.20]
    rows[1, :5] = [0.3, 0.52, 0.52, 0.20, 0.20]

    with np.errstate(all='raise'):
        merged = _nms_within(rows, iou_threshold=1.0)

    # Nothing overlaps enough to blend, so both survive untouched.
    assert len(merged) == 2
    assert merged[0, 1] == pytest.approx(0.50)
    assert merged[1, 1] == pytest.approx(0.52)


def test_weighted_nms_terminates_on_zero_area_box():
    """A degenerate box scores an IoU of 0 against itself."""
    columns = 5 + 2 * 6
    rows = np.zeros((1, columns))
    rows[0, :5] = [0.9, 0.50, 0.50, 0.0, 0.0]

    with np.errstate(all='raise'):
        merged = _nms_within(rows, iou_threshold=0.3)

    assert len(merged) == 1
    assert merged[0, :5] == pytest.approx([0.9, 0.50, 0.50, 0.0, 0.0])


def test_analyzer_disables_recognition(blazeface_model, face_image):
    """The regression test for the alignment seam.

    An unalignable detector must not crash the analyzer, must not silently
    produce broken embeddings, and must say so once rather than per face.
    """
    from uniface.analyzer import FaceAnalyzer
    from uniface.recognition import ArcFace

    analyzer = FaceAnalyzer(detector=blazeface_model, recognizer=ArcFace())

    assert analyzer.recognizer is None

    faces = analyzer.analyze(face_image)
    assert all(face.embedding is None for face in faces)
