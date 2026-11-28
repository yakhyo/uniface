# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import numpy as np

from uniface.tracking import BYTETracker
from uniface.tracking.bytetrack.basetrack import BaseTrack
from uniface.tracking.bytetrack.kalman import KalmanFilter


def test_multiple_bytetrackers_do_not_collide_after_reset():
    """Resetting one BYTETracker must not reset IDs or drop new faces in another active tracker."""
    t1 = BYTETracker()
    t2 = BYTETracker()

    r1 = t1.update(np.array([[100.0, 100.0, 200.0, 200.0, 0.9]]))
    r2 = t2.update(np.array([[100.0, 100.0, 200.0, 200.0, 0.9]]))
    assert r1[:, 4].tolist() == [1.0]
    assert r2[:, 4].tolist() == [1.0]

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
    """Short-lived detections must not grow removed_stracks without bound."""
    tracker = BYTETracker(track_thresh=0.5, track_buffer=1)

    for frame_idx in range(1200):
        x = float((frame_idx % 20) * 50)
        tracker.update(np.array([[x, 0.0, x + 20.0, 20.0, 0.8]]))
        tracker.update(np.empty((0, 5)))

    assert len(tracker.removed_stracks) <= 1000


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
