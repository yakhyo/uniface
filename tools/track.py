# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Face tracking on video files using ByteTrack.

Usage:
    python tools/track.py --source video.mp4
    python tools/track.py --source video.mp4 --output outputs/tracked.mp4
    python tools/track.py --source 0  # webcam
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from _common import VIDEO_EXTENSIONS
import cv2
import numpy as np
from tqdm import tqdm

from uniface.common import xyxy_to_cxcywh
from uniface.detection import SCRFD, RetinaFace
from uniface.draw import draw_tracks
from uniface.tracking import BYTETracker


def _assign_track_ids(faces, tracks) -> list:
    """Match tracker outputs back to Face objects by center distance."""
    if len(tracks) == 0 or len(faces) == 0:
        return []

    face_bboxes = np.array([f.bbox for f in faces], dtype=np.float32)
    track_ids = tracks[:, 4].astype(int)

    face_centers = xyxy_to_cxcywh(face_bboxes)[:, :2]  # (N, 2) -> [cx, cy]
    track_centers = xyxy_to_cxcywh(tracks[:, :4])[:, :2]  # (M, 2) -> [cx, cy]

    for ti in range(len(tracks)):
        dists = (track_centers[ti, 0] - face_centers[:, 0]) ** 2 + (track_centers[ti, 1] - face_centers[:, 1]) ** 2
        faces[int(np.argmin(dists))].track_id = track_ids[ti]

    return [f for f in faces if f.track_id is not None]


def process_video(
    detector,
    tracker: BYTETracker,
    input_path: str,
    output_path: str,
    threshold: float = 0.5,
    show_preview: bool = False,
):
    """Process a video file with face tracking."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{input_path}'")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f'Input: {input_path} ({width}x{height}, {fps:.1f} fps, {total_frames} frames)')
    print(f'Output: {output_path}')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Cannot create output video '{output_path}'")
        cap.release()
        return

    frame_count = 0
    total_tracks = 0

    for _ in tqdm(range(total_frames), desc='Tracking', unit='frames'):
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detect faces
        faces = detector.detect(frame)
        dets = np.array([[*f.bbox, f.confidence] for f in faces if f.confidence >= threshold])
        dets = dets if len(dets) > 0 else np.empty((0, 5))

        # Update tracker
        tracks = tracker.update(dets)
        tracked_faces = _assign_track_ids(faces, tracks)
        total_tracks += len(tracked_faces)

        # Draw tracked faces
        draw_tracks(image=frame, faces=tracked_faces)

        cv2.putText(frame, f'Tracks: {len(tracked_faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)

        if show_preview:
            cv2.imshow("Tracking - Press 'q' to cancel", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('\nCancelled by user')
                break

    cap.release()
    out.release()
    if show_preview:
        cv2.destroyAllWindows()

    avg_tracks = total_tracks / frame_count if frame_count > 0 else 0
    print(f'\nDone! {frame_count} frames, {total_tracks} tracks ({avg_tracks:.1f} avg/frame)')
    print(f'Saved: {output_path}')


def run_camera(
    detector,
    tracker: BYTETracker,
    camera_id: int = 0,
    threshold: float = 0.5,
):
    """Run real-time face tracking on webcam."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f'Cannot open camera {camera_id}')
        return

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        # Detect faces
        faces = detector.detect(frame)
        dets = np.array([[*f.bbox, f.confidence] for f in faces if f.confidence >= threshold])
        dets = dets if len(dets) > 0 else np.empty((0, 5))

        # Update tracker
        tracks = tracker.update(dets)
        tracked_faces = _assign_track_ids(faces, tracks)

        # Draw tracked faces
        draw_tracks(image=frame, faces=tracked_faces)

        cv2.putText(frame, f'Tracks: {len(tracked_faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Face tracking on video using ByteTrack')
    parser.add_argument('--source', type=str, required=True, help='Video path or camera ID (0, 1, ...)')
    parser.add_argument('--output', type=str, default=None, help='Output video path')
    parser.add_argument('--detector', type=str, default='scrfd', choices=['retinaface', 'scrfd'])
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--track-buffer', type=int, default=30, help='Max frames to keep lost tracks')
    parser.add_argument('--preview', action='store_true', help='Show live preview')
    parser.add_argument('--save-dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()

    detector = RetinaFace() if args.detector == 'retinaface' else SCRFD()
    tracker = BYTETracker(track_thresh=args.threshold, track_buffer=args.track_buffer)

    if args.source.isdigit():
        run_camera(detector, tracker, int(args.source), args.threshold)
    else:
        if not os.path.exists(args.source):
            print(f'Error: Video not found: {args.source}')
            return

        ext = Path(args.source).suffix.lower()
        if ext not in VIDEO_EXTENSIONS:
            print(f"Error: Unsupported format '{ext}'. Supported: {VIDEO_EXTENSIONS}")
            return

        if args.output:
            output_path = args.output
        else:
            os.makedirs(args.save_dir, exist_ok=True)
            output_path = os.path.join(args.save_dir, f'{Path(args.source).stem}_tracked.mp4')

        process_video(detector, tracker, args.source, output_path, args.threshold, args.preview)


if __name__ == '__main__':
    main()
