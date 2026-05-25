# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Face Image Quality Assessment.

Usage:
    python tools/quality.py --source path/to/image.jpg
    python tools/quality.py --source path/to/video.mp4
    python tools/quality.py --source 0  # webcam
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from _common import get_source_type
import cv2

from uniface.constants import EDifFIQAWeights
from uniface.detection import SCRFD
from uniface.draw import draw_quality_score
from uniface.quality import EDifFIQA

VARIANT_MAP = {
    't': EDifFIQAWeights.T,
    's': EDifFIQAWeights.S,
    'm': EDifFIQAWeights.M,
    'l': EDifFIQAWeights.L,
}


def process_image(detector, quality, image_path: str, save_dir: str = 'outputs') -> None:
    """Score every detected face in an image and save an annotated copy."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from '{image_path}'")
        return

    faces = detector.detect(image)
    print(f'Detected {len(faces)} face(s)')

    if not faces:
        print('No faces detected in the image.')
        return

    for i, face in enumerate(faces, 1):
        result = quality.predict(image, face.landmarks)
        print(f'  Face {i}: quality={result.score:.4f}')
        draw_quality_score(image, face.bbox, result.score)

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{Path(image_path).stem}_quality.jpg')
    cv2.imwrite(output_path, image)
    print(f'Output saved: {output_path}')


def process_video(detector, quality, video_path: str, save_dir: str = 'outputs') -> None:
    """Score faces frame-by-frame in a video and save an annotated copy."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{Path(video_path).stem}_quality.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f'Processing video: {video_path} ({total_frames} frames)')
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        faces = detector.detect(frame)

        for face in faces:
            result = quality.predict(frame, face.landmarks)
            draw_quality_score(frame, face.bbox, result.score)

        out.write(frame)

        if frame_count % 100 == 0:
            print(f'  Processed {frame_count}/{total_frames} frames...')

    cap.release()
    out.release()
    print(f'Done! Output saved: {output_path}')


def run_camera(detector, quality, camera_id: int = 0) -> None:
    """Run real-time quality assessment on webcam."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f'Cannot open camera {camera_id}')
        return

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        faces = detector.detect(frame)

        for face in faces:
            result = quality.predict(frame, face.landmarks)
            draw_quality_score(frame, face.bbox, result.score)

        cv2.imshow('Face Quality Assessment', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Face Image Quality Assessment (eDifFIQA)')
    parser.add_argument('--source', type=str, required=True, help='Image/video path or camera ID (0, 1, ...)')
    parser.add_argument(
        '--variant',
        type=str,
        default='t',
        choices=['t', 's', 'm', 'l'],
        help='eDifFIQA model variant (default: t)',
    )
    parser.add_argument('--save-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument(
        '--detector-conf',
        type=float,
        default=0.3,
        help='SCRFD confidence threshold (default: 0.3, lower than the SCRFD default of 0.5)',
    )
    args = parser.parse_args()

    model_name = VARIANT_MAP[args.variant]

    print(f'Initializing models (SCRFD + eDifFIQA-{args.variant.upper()})...')
    # Lower the default SCRFD threshold (0.5 -> 0.3): quality scoring wants
    # to see every plausible face, including the low-confidence ones that
    # are precisely what the quality model is meant to flag.
    detector = SCRFD(confidence_threshold=args.detector_conf)
    quality = EDifFIQA(model_name=model_name)

    source_type = get_source_type(args.source)

    if source_type == 'camera':
        run_camera(detector, quality, int(args.source))
    elif source_type == 'image':
        if not os.path.exists(args.source):
            print(f'Error: Image not found: {args.source}')
            return
        process_image(detector, quality, args.source, args.save_dir)
    elif source_type == 'video':
        if not os.path.exists(args.source):
            print(f'Error: Video not found: {args.source}')
            return
        process_video(detector, quality, args.source, args.save_dir)
    else:
        print(f"Error: Unknown source type for '{args.source}'")
        print('Supported formats: images (.jpg, .png, ...), videos (.mp4, .avi, ...), or camera ID (0, 1, ...)')


if __name__ == '__main__':
    main()
