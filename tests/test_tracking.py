# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import numpy as np

from uniface.tracking import BYTETracker


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


def test_lost_track_is_refound_on_the_frame_after_timeout():
    """A timed-out lost track stays matchable for one more frame, as in upstream ByteTrack."""
    tracker = BYTETracker(track_buffer=5)
    face = np.array([[100.0, 100.0, 200.0, 200.0, 0.9]])

    first_ids = tracker.update(face)[:, 4].tolist()
    for frame in range(2, 10):
        tracker.update(face if frame <= 3 else np.empty((0, 5)))

    assert tracker.update(face)[:, 4].tolist() == first_ids


def test_removed_stracks_is_bounded():
    """Short-lived detections must not grow removed_stracks without bound."""
    tracker = BYTETracker(track_thresh=0.5, track_buffer=1)

    for frame_idx in range(1200):
        x = float((frame_idx % 20) * 50)
        tracker.update(np.array([[x, 0.0, x + 20.0, 20.0, 0.8]]))
        tracker.update(np.empty((0, 5)))

    assert len(tracker.removed_stracks) <= 1000
