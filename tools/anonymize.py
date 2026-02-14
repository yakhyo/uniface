# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Face anonymization/blurring for privacy.

Usage:
    python tools/anonymize.py --source path/to/image.jpg --method pixelate
    python tools/anonymize.py --source path/to/video.mp4 --method gaussian
    python tools/anonymize.py --source 0 --method pixelate  # webcam
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from _common import get_source_type
import cv2

from uniface.detection import RetinaFace
from uniface.privacy import BlurFace


def process_image(
    detector,
    blurrer: BlurFace,
    image_path: str,
    save_dir: str = 'outputs',
    show_detections: bool = False,
):
    """Process a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from '{image_path}'")
        return

    faces = detector.detect(image)
    print(f'Detected {len(faces)} face(s)')

    if show_detections and faces:
        from uniface.draw import draw_detections

        preview = image.copy()
        bboxes = [face.bbox for face in faces]
        scores = [face.confidence for face in faces]
        landmarks = [face.landmarks for face in faces]
        draw_detections(preview, bboxes, scores, landmarks)

        cv2.imshow('Detections (Press any key to continue)', preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if faces:
        anonymized = blurrer.anonymize(image, faces)
    else:
        anonymized = image

    os.makedirs(save_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(save_dir, f'{basename}_anonymized.jpg')
    cv2.imwrite(output_path, anonymized)
    print(f'Output saved: {output_path}')


def process_video(
    detector,
    blurrer: BlurFace,
    video_path: str,
    save_dir: str = 'outputs',
):
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
    output_path = os.path.join(save_dir, f'{Path(video_path).stem}_anonymized.mp4')
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

        if faces:
            frame = blurrer.anonymize(frame, faces, inplace=True)

        out.write(frame)

        if frame_count % 100 == 0:
            print(f'  Processed {frame_count}/{total_frames} frames...')

    cap.release()
    out.release()
    print(f'Done! Output saved: {output_path}')


def run_camera(detector, blurrer: BlurFace, camera_id: int = 0):
    """Run real-time anonymization on webcam."""
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
        if faces:
            frame = blurrer.anonymize(frame, faces, inplace=True)

        cv2.putText(
            frame,
            f'Faces blurred: {len(faces)} | Method: {blurrer.method}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow('Face Anonymization (Press q to quit)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Face anonymization using various blur methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Anonymize image with pixelation (default)
  python tools/anonymize.py --source photo.jpg

  # Use Gaussian blur with custom strength
  python tools/anonymize.py --source photo.jpg --method gaussian --blur-strength 5.0

  # Real-time webcam anonymization
  python tools/anonymize.py --source 0 --method pixelate

  # Black boxes for maximum privacy
  python tools/anonymize.py --source photo.jpg --method blackout

  # Custom pixelation intensity
  python tools/anonymize.py --source photo.jpg --method pixelate --pixel-blocks 5
        """,
    )

    # Input/output
    parser.add_argument('--source', type=str, required=True, help='Image/video path or camera ID (0, 1, ...)')
    parser.add_argument('--save-dir', type=str, default='outputs', help='Output directory')

    # Blur method
    parser.add_argument(
        '--method',
        type=str,
        default='pixelate',
        choices=['gaussian', 'pixelate', 'blackout', 'elliptical', 'median'],
        help='Blur method (default: pixelate)',
    )

    # Method-specific parameters
    parser.add_argument(
        '--blur-strength',
        type=float,
        default=3.0,
        help='Blur strength for gaussian/elliptical/median (default: 3.0)',
    )
    parser.add_argument(
        '--pixel-blocks',
        type=int,
        default=20,
        help='Number of pixel blocks for pixelate (default: 20, lower=more pixelated)',
    )
    parser.add_argument(
        '--color',
        type=str,
        default='0,0,0',
        help='Fill color for blackout as R,G,B (default: 0,0,0 for black)',
    )
    parser.add_argument('--margin', type=int, default=20, help='Margin for elliptical blur (default: 20)')

    # Detection
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.5,
        help='Detection confidence threshold (default: 0.5)',
    )

    # Visualization
    parser.add_argument(
        '--show-detections',
        action='store_true',
        help='Show detection boxes before blurring (image mode only)',
    )

    args = parser.parse_args()

    # Parse color
    color_values = [int(x) for x in args.color.split(',')]
    if len(color_values) != 3:
        parser.error('--color must be in format R,G,B (e.g., 0,0,0)')
    color = tuple(color_values)

    # Initialize detector
    print(f'Initializing face detector (confidence_threshold={args.confidence_threshold})...')
    detector = RetinaFace(confidence_threshold=args.confidence_threshold)

    # Initialize blurrer
    print(f'Initializing blur method: {args.method}')
    blurrer = BlurFace(
        method=args.method,
        blur_strength=args.blur_strength,
        pixel_blocks=args.pixel_blocks,
        color=color,
        margin=args.margin,
    )

    source_type = get_source_type(args.source)

    if source_type == 'camera':
        run_camera(detector, blurrer, int(args.source))
    elif source_type == 'image':
        if not os.path.exists(args.source):
            print(f'Error: Image not found: {args.source}')
            return
        process_image(detector, blurrer, args.source, args.save_dir, args.show_detections)
    elif source_type == 'video':
        if not os.path.exists(args.source):
            print(f'Error: Video not found: {args.source}')
            return
        process_video(detector, blurrer, args.source, args.save_dir)
    else:
        print(f"Error: Unknown source type for '{args.source}'")
        print('Supported formats: images (.jpg, .png, ...), videos (.mp4, .avi, ...), or camera ID (0, 1, ...)')


if __name__ == '__main__':
    main()
