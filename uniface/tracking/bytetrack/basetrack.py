# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from collections import OrderedDict

import numpy as np


class TrackState:
    """Track state enumeration."""

    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    """Base class for tracked objects."""

    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()  # noqa: RUF012
    features = []  # noqa: RUF012
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    location = (np.inf, np.inf)

    @property
    def end_frame(self) -> int:
        """Return the last frame ID."""
        return self.frame_id

    @staticmethod
    def next_id() -> int:
        """Generate next unique track ID."""
        BaseTrack._count += 1
        return BaseTrack._count

    @staticmethod
    def reset_id() -> None:
        """Reset the ID counter."""
        BaseTrack._count = 0

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self) -> None:
        """Mark track as lost."""
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        """Mark track as removed."""
        self.state = TrackState.Removed
