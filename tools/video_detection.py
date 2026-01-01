# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Face detection on video files with progress tracking.

Usage:
    python tools/video_detection.py --source video.mp4
    python tools/video_detection.py --source video.mp4 --output output.mp4
    python tools/video_detection.py --source 0  # webcam
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
from tqdm import tqdm

from uniface import SCRFD, RetinaFace
from uniface.visualization import draw_detections

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}


def get_source_type(source: str) -> str:
    """Determine if source is image, video, or camera."""
    if source.isdigit():
        return 'camera'
    path = Path(source)
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return 'image'
    elif suffix in VIDEO_EXTENSIONS:
        return 'video'
    else:
        return 'unknown'


def process_video(
    detector,
    input_path: str,
    output_path: str,
    threshold: float = 0.6,
    show_preview: bool = False,
):
    """Process a video file with progress bar."""
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
    total_faces = 0

    for _ in tqdm(range(total_frames), desc='Processing', unit='frames'):
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        faces = detector.detect(frame)
        total_faces += len(faces)

        bboxes = [f.bbox for f in faces]
        scores = [f.confidence for f in faces]
        landmarks = [f.landmarks for f in faces]
        draw_detections(
            image=frame, bboxes=bboxes, scores=scores, landmarks=landmarks, vis_threshold=threshold, fancy_bbox=True
        )

        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)

        if show_preview:
            cv2.imshow("Processing - Press 'q' to cancel", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('\nCancelled by user')
                break

    cap.release()
    out.release()
    if show_preview:
        cv2.destroyAllWindows()

    avg_faces = total_faces / frame_count if frame_count > 0 else 0
    print(f'\nDone! {frame_count} frames, {total_faces} faces ({avg_faces:.1f} avg/frame)')
    print(f'Saved: {output_path}')


def run_camera(detector, camera_id: int = 0, threshold: float = 0.6):
    """Run real-time detection on webcam."""
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

        faces = detector.detect(frame)

        bboxes = [f.bbox for f in faces]
        scores = [f.confidence for f in faces]
        landmarks = [f.landmarks for f in faces]
        draw_detections(
            image=frame, bboxes=bboxes, scores=scores, landmarks=landmarks, vis_threshold=threshold, fancy_bbox=True
        )

        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Process video with face detection')
    parser.add_argument('--source', type=str, required=True, help='Video path or camera ID (0, 1, ...)')
    parser.add_argument('--output', type=str, default=None, help='Output video path (auto-generated if not specified)')
    parser.add_argument('--detector', type=str, default='retinaface', choices=['retinaface', 'scrfd'])
    parser.add_argument('--threshold', type=float, default=0.6, help='Visualization threshold')
    parser.add_argument('--preview', action='store_true', help='Show live preview')
    parser.add_argument('--save-dir', type=str, default='outputs', help='Output directory (if --output not specified)')
    args = parser.parse_args()

    detector = RetinaFace() if args.detector == 'retinaface' else SCRFD()

    source_type = get_source_type(args.source)

    if source_type == 'camera':
        run_camera(detector, int(args.source), args.threshold)
    elif source_type == 'video':
        if not os.path.exists(args.source):
            print(f'Error: Video not found: {args.source}')
            return

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            os.makedirs(args.save_dir, exist_ok=True)
            output_path = os.path.join(args.save_dir, f'{Path(args.source).stem}_detected.mp4')

        process_video(detector, args.source, output_path, args.threshold, args.preview)
    else:
        print(f"Error: Unknown source type for '{args.source}'")
        print('Supported formats: videos (.mp4, .avi, ...) or camera ID (0, 1, ...)')


if __name__ == '__main__':
    main()
