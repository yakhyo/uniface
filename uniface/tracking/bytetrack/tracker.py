# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import numpy as np

from . import matching
from .basetrack import BaseTrack, TrackState
from .kalman import KalmanFilter


class STrack(BaseTrack):
    """Single object track using Kalman filter."""

    shared_kalman = KalmanFilter()

    def __init__(self, tlwh: np.ndarray, score: float) -> None:
        """Initialize STrack.

        Args:
            tlwh: Bounding box in [x, y, w, h] format (top-left).
            score: Detection confidence score.
        """
        super().__init__()
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0

    def predict(self) -> None:
        """Predict next state using Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks: list) -> None:
        """Predict states for multiple tracks (vectorized).

        Args:
            stracks: List of STrack objects.
        """
        if len(stracks) == 0:
            return

        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])

        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0

        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)

        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance, strict=False)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def activate(self, kalman_filter: KalmanFilter, frame_id: int) -> None:
        """Start a new tracklet.

        Args:
            kalman_filter: Kalman filter instance.
            frame_id: Current frame number.
        """
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self._tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False) -> None:
        """Reactivate a lost track.

        Args:
            new_track: New detection to reactivate with.
            frame_id: Current frame number.
            new_id: Whether to assign a new track ID.
        """
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self._tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track: STrack, frame_id: int) -> None:
        """Update matched track with new detection.

        Args:
            new_track: Matched detection.
            frame_id: Current frame number.
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self._tlwh_to_xyah(new_track.tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    @property
    def tlwh(self) -> np.ndarray:
        """Get bounding box in [x, y, w, h] format (top-left)."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        """Get bounding box in [x1, y1, x2, y2] format."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def _tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """Convert [x, y, w, h] to [cx, cy, aspect_ratio, height]."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def _tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    def __repr__(self) -> str:
        return f'OT_{self.track_id}_({self.start_frame}-{self.end_frame})'


class BYTETracker:
    """ByteTrack multi-object tracker.

    Uses two-stage association to match both high and low confidence detections.

    Args:
        track_thresh: High confidence detection threshold.
        track_buffer: Maximum frames to keep lost tracks.
        match_thresh: IoU threshold for first association.
        low_thresh: Low confidence detection threshold.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        low_thresh: float = 0.1,
    ) -> None:
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.low_thresh = low_thresh
        self.buffer_size = track_buffer

        self.frame_id = 0
        self.kalman_filter = KalmanFilter()
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []

    def reset(self) -> None:
        """Reset tracker state."""
        self.frame_id = 0
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        BaseTrack.reset_id()

    def update(self, detections: np.ndarray) -> np.ndarray:
        """Update tracker with new detections.

        Args:
            detections: Array of shape (N, 5) with [x1, y1, x2, y2, score].

        Returns:
            Array of shape (M, 5) with [x1, y1, x2, y2, track_id].
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(detections) == 0:
            detections = np.empty((0, 5))

        scores = detections[:, 4]
        bboxes = detections[:, :4]

        # Split detections into high and low confidence
        high_inds = scores >= self.track_thresh
        low_inds = (scores > self.low_thresh) & (scores < self.track_thresh)

        dets_high = bboxes[high_inds]
        scores_high = scores[high_inds]
        dets_low = bboxes[low_inds]
        scores_low = scores[low_inds]

        # Create STrack objects
        detections_high = [
            STrack(STrack._tlbr_to_tlwh(tlbr), s) for tlbr, s in zip(dets_high, scores_high, strict=False)
        ]
        detections_low = [STrack(STrack._tlbr_to_tlwh(tlbr), s) for tlbr, s in zip(dets_low, scores_low, strict=False)]

        # Separate confirmed and unconfirmed tracks
        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        tracked_stracks = [t for t in self.tracked_stracks if t.is_activated]

        # Step 1: First association with high confidence detections
        strack_pool = _joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)

        dists = matching.iou_distance(strack_pool, detections_high)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_high[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Step 2: Second association with low confidence detections
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_low)
        matches, u_track_second, _ = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Mark unmatched tracks as lost
        for it in u_track_second:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Step 3: Associate unconfirmed tracks
        detections_remain = [detections_high[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections_remain)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections_remain[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Step 4: Initialize new tracks
        for inew in u_detection:
            track = detections_remain[inew]
            if track.score < self.track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # Step 5: Remove old lost tracks
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.buffer_size:
                track.mark_removed()
                removed_stracks.append(track)

        # Update track lists
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = _joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = _joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = _sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = _sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = _remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)

        # Output confirmed tracks
        outputs = []
        for track in self.tracked_stracks:
            if track.is_activated:
                outputs.append(np.concatenate([track.tlbr, [track.track_id]]))

        return np.array(outputs) if outputs else np.empty((0, 5))


def _joint_stracks(tlista: list, tlistb: list) -> list:
    """Merge two track lists without duplicates."""
    exists = {t.track_id: 1 for t in tlista}
    res = list(tlista)
    for t in tlistb:
        if not exists.get(t.track_id, 0):
            exists[t.track_id] = 1
            res.append(t)
    return res


def _sub_stracks(tlista: list, tlistb: list) -> list:
    """Remove tracks in tlistb from tlista."""
    track_ids_b = {t.track_id for t in tlistb}
    return [t for t in tlista if t.track_id not in track_ids_b]


def _remove_duplicate_stracks(stracksa: list, stracksb: list) -> tuple[list, list]:
    """Remove duplicate tracks based on IoU."""
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = [], []

    for p, q in zip(*pairs, strict=False):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)

    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
