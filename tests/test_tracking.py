# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import numpy as np

from uniface.tracking import BYTETracker
from uniface.tracking.bytetrack.basetrack import BaseTrack
from uniface.tracking.bytetrack.kalman import KalmanFilter


def test_bytetracker_two_stage_low_confidence_association():
    """Low-confidence detections (between low_thresh and track_thresh) keep an active track alive."""
    tracker = BYTETracker(track_thresh=0.5, low_thresh=0.1, track_buffer=5)

    # Frame 1: high-confidence detection activates track 1
    out1 = tracker.update(np.array([[100.0, 100.0, 200.0, 200.0, 0.9]]))
    assert len(out1) == 1
    assert out1[0, 4] == 1

    # Frame 2: confidence dips to 0.25 (below track_thresh=0.5, above low_thresh=0.1)
    out2 = tracker.update(np.array([[102.0, 101.0, 202.0, 201.0, 0.25]]))
    assert len(out2) == 1
    assert out2[0, 4] == 1


def test_multiple_bytetrackers_do_not_collide_after_reset():
    """Resetting one BYTETracker must not reset IDs or drop new faces in another active tracker."""
    t1 = BYTETracker()
    t2 = BYTETracker()

    r1 = t1.update(np.array([[100.0, 100.0, 200.0, 200.0, 0.9]]))
    r2 = t2.update(np.array([[100.0, 100.0, 200.0, 200.0, 0.9]]))
    assert r1[:, 4].tolist() == [1.0]
    assert r2[:, 4].tolist() == [1.0]

    # Reset t1 while t2 continues tracking and sees two new faces
    t1.reset()
    three_faces = np.array(
        [
            [100.0, 100.0, 200.0, 200.0, 0.9],
            [300.0, 300.0, 400.0, 400.0, 0.9],
            [500.0, 500.0, 600.0, 600.0, 0.9],
        ]
    )
    t2.update(three_faces)
    out = t2.update(three_faces)

    assert len(out) == 3
    assert len(set(out[:, 4].tolist())) == 3


def test_basetrack_mutable_containers_are_per_instance():
    a = BaseTrack()
    b = BaseTrack()

    a.features.append('feat_a')
    a.history[1] = 'frame_1'

    assert b.features == []
    assert list(b.history) == []


def test_removed_stracks_is_bounded():
    """Transient unconfirmed detections must not grow removed_stracks without bound."""
    tracker = BYTETracker(track_thresh=0.5, track_buffer=1)

    for frame_idx in range(1200):
        # Single-frame unconfirmed detection at a shifting location, followed by an empty frame
        x = float((frame_idx % 20) * 50)
        tracker.update(np.array([[x, 0.0, x + 20.0, 20.0, 0.8]]))
        tracker.update(np.empty((0, 5)))

    assert len(tracker.removed_stracks) <= tracker._MAX_REMOVED_TRACKS


def test_kalman_multi_predict_matches_single_predict():
    kf = KalmanFilter()
    rng = np.random.default_rng(42)

    means, covs = [], []
    expected_means, expected_covs = [], []
    for _ in range(10):
        measurement = rng.uniform(50.0, 250.0, size=4)
        measurement[2] = rng.uniform(0.6, 1.4)
        m, c = kf.initiate(measurement)
        means.append(m)
        covs.append(c)
        pm, pc = kf.predict(m.copy(), c.copy())
        expected_means.append(pm)
        expected_covs.append(pc)

    actual_means, actual_covs = kf.multi_predict(np.asarray(means), np.asarray(covs))
    assert np.allclose(actual_means, np.asarray(expected_means), atol=1e-12)
    assert np.allclose(actual_covs, np.asarray(expected_covs), atol=1e-12)


def test_track_tool_assign_track_ids_is_one_to_one():
    """An unmatched/occluded track must not overwrite another detected face's track_id."""
    from pathlib import Path
    import sys

    tools_dir = str(Path(__file__).resolve().parent.parent / 'tools')
    sys.path.insert(0, tools_dir)
    try:
        from track import _assign_track_ids
    finally:
        sys.path.remove(tools_dir)

    from uniface.types import Face

    face1 = Face(bbox=np.array([100.0, 100.0, 200.0, 200.0]), confidence=0.9, landmarks=np.zeros((5, 2)))
    # Track 1 matches face1; Track 2 is at (500, 500) whose face is temporarily occluded
    tracks = np.array(
        [
            [100.0, 100.0, 200.0, 200.0, 1.0],
            [500.0, 500.0, 600.0, 600.0, 2.0],
        ]
    )

    assigned = _assign_track_ids([face1], tracks)
    assert len(assigned) == 1
    assert assigned[0].track_id == 1
