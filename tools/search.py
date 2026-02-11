# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Real-time face search: match faces against a reference image.

Usage:
    python tools/search.py --reference person.jpg --source 0  # webcam
    python tools/search.py --reference person.jpg --source video.mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from _common import get_source_type
import cv2
import numpy as np

from uniface.detection import SCRFD, RetinaFace
from uniface.face_utils import compute_similarity
from uniface.recognition import ArcFace, MobileFace, SphereFace


def get_recognizer(name: str):
    """Get recognizer by name."""
    if name == 'arcface':
        return ArcFace()
    elif name == 'mobileface':
        return MobileFace()
    else:
        return SphereFace()


def extract_reference_embedding(detector, recognizer, image_path: str) -> np.ndarray:
    """Extract embedding from reference image."""
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f'Failed to load image: {image_path}')

    faces = detector.detect(image)
    if not faces:
        raise RuntimeError('No faces found in reference image.')

    landmarks = faces[0].landmarks
    return recognizer.get_normalized_embedding(image, landmarks)


def process_frame(frame, detector, recognizer, ref_embedding: np.ndarray, threshold: float = 0.4):
    """Process a single frame and return annotated frame."""
    faces = detector.detect(frame)

    for face in faces:
        bbox = face.bbox
        landmarks = face.landmarks
        x1, y1, x2, y2 = map(int, bbox)

        embedding = recognizer.get_normalized_embedding(frame, landmarks)
        sim = compute_similarity(ref_embedding, embedding)

        label = f'Match ({sim:.2f})' if sim > threshold else f'Unknown ({sim:.2f})'
        color = (0, 255, 0) if sim > threshold else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


def process_video(detector, recognizer, ref_embedding: np.ndarray, video_path: str, save_dir: str, threshold: float):
    """Process a video file."""
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
    """Run real-time face search on webcam."""
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

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Face search using a reference image')
    parser.add_argument('--reference', type=str, required=True, help='Reference face image')
    parser.add_argument('--source', type=str, required=True, help='Video path or camera ID (0, 1, ...)')
    parser.add_argument('--threshold', type=float, default=0.4, help='Match threshold')
    parser.add_argument('--detector', type=str, default='scrfd', choices=['retinaface', 'scrfd'])
    parser.add_argument(
        '--recognizer',
        type=str,
        default='arcface',
        choices=['arcface', 'mobileface', 'sphereface'],
    )
    parser.add_argument('--save-dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()

    if not os.path.exists(args.reference):
        print(f'Error: Reference image not found: {args.reference}')
        return

    detector = RetinaFace() if args.detector == 'retinaface' else SCRFD()
    recognizer = get_recognizer(args.recognizer)

    print(f'Loading reference: {args.reference}')
    ref_embedding = extract_reference_embedding(detector, recognizer, args.reference)

    source_type = get_source_type(args.source)

    if source_type == 'camera':
        run_camera(detector, recognizer, ref_embedding, int(args.source), args.threshold)
    elif source_type == 'video':
        if not os.path.exists(args.source):
            print(f'Error: Video not found: {args.source}')
            return
        process_video(detector, recognizer, ref_embedding, args.source, args.save_dir, args.threshold)
    else:
        print(f"Error: Source must be a video file or camera ID, not '{args.source}'")
        print('Supported formats: videos (.mp4, .avi, ...) or camera ID (0, 1, ...)')


if __name__ == '__main__':
    main()
