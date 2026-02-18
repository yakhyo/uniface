# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Single-reference face search on video or webcam.

Given a reference face image, detects faces in the source and shows
whether each face matches the reference.

Usage:
    python tools/search.py --reference ref.jpg --source video.mp4
    python tools/search.py --reference ref.jpg --source 0  # webcam
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from _common import get_source_type
import cv2
import numpy as np

from uniface import create_detector, create_recognizer
from uniface.draw import draw_corner_bbox, draw_text_label
from uniface.face_utils import compute_similarity


def extract_reference_embedding(detector, recognizer, image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f'Failed to load image: {image_path}')

    faces = detector.detect(image)
    if not faces:
        raise RuntimeError('No faces found in reference image.')

    return recognizer.get_normalized_embedding(image, faces[0].landmarks)


def _draw_face(image, bbox, text: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = map(int, bbox[:4])
    thickness = max(round(sum(image.shape[:2]) / 2 * 0.003), 2)
    font_scale = max(0.4, min(0.7, (y2 - y1) / 200))
    draw_corner_bbox(image, (x1, y1, x2, y2), color=color, thickness=thickness)
    draw_text_label(image, text, x1, y1, bg_color=color, font_scale=font_scale)


def process_frame(frame, detector, recognizer, ref_embedding: np.ndarray, threshold: float = 0.4):
    faces = detector.detect(frame)

    for face in faces:
        embedding = recognizer.get_normalized_embedding(frame, face.landmarks)
        sim = compute_similarity(ref_embedding, embedding)

        text = f'Match ({sim:.2f})' if sim > threshold else f'Unknown ({sim:.2f})'
        color = (0, 255, 0) if sim > threshold else (0, 0, 255)
        _draw_face(frame, face.bbox, text, color)

    return frame


def process_video(
    detector, recognizer, video_path: str, save_dir: str, ref_embedding: np.ndarray, threshold: float = 0.4
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{Path(video_path).stem}_search.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f'Processing video: {video_path} ({total_frames} frames)')
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = process_frame(frame, detector, recognizer, ref_embedding, threshold)
        out.write(frame)

        if frame_count % 100 == 0:
            print(f'  Processed {frame_count}/{total_frames} frames...')

    cap.release()
    out.release()
    print(f'Done! Output saved: {output_path}')


def run_camera(detector, recognizer, ref_embedding: np.ndarray, camera_id: int = 0, threshold: float = 0.4):
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

        frame = process_frame(frame, detector, recognizer, ref_embedding, threshold)

        cv2.imshow('Face Search', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Single-reference face search')
    parser.add_argument('--reference', type=str, required=True, help='Reference face image')
    parser.add_argument('--source', type=str, required=True, help='Video path or camera ID')
    parser.add_argument('--threshold', type=float, default=0.4, help='Similarity threshold')
    parser.add_argument('--save-dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()

    if not os.path.exists(args.reference):
        print(f'Error: Reference image not found: {args.reference}')
        return

    detector = create_detector()
    recognizer = create_recognizer()

    print(f'Loading reference: {args.reference}')
    ref_embedding = extract_reference_embedding(detector, recognizer, args.reference)

    source_type = get_source_type(args.source)

    if source_type == 'camera':
        run_camera(detector, recognizer, ref_embedding, int(args.source), args.threshold)
    elif source_type == 'video':
        if not os.path.exists(args.source):
            print(f'Error: Video not found: {args.source}')
            return
        process_video(detector, recognizer, args.source, args.save_dir, ref_embedding, args.threshold)
    else:
        print(f"Error: Source must be a video file or camera ID, not '{args.source}'")


if __name__ == '__main__':
    main()
