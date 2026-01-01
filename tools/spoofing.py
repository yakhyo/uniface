# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Face Anti-Spoofing Detection.

Usage:
    python tools/spoofing.py --source path/to/image.jpg
    python tools/spoofing.py --source path/to/video.mp4
    python tools/spoofing.py --source 0  # webcam
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from uniface import RetinaFace
from uniface.constants import MiniFASNetWeights
from uniface.spoofing import create_spoofer

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


def draw_spoofing_result(
    image: np.ndarray,
    bbox: list,
    is_real: bool,
    confidence: float,
    thickness: int = 2,
) -> None:
    """Draw bounding box with anti-spoofing result.

    Args:
        image: Input image to draw on.
        bbox: Bounding box in [x1, y1, x2, y2] format.
        is_real: True if real face, False if fake.
        confidence: Confidence score (0.0 to 1.0).
        thickness: Line thickness for bounding box.
    """
    x1, y1, x2, y2 = map(int, bbox[:4])

    color = (0, 255, 0) if is_real else (0, 0, 255)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    label = 'Real' if is_real else 'Fake'
    text = f'{label}: {confidence:.1%}'

    (tw, th), _baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    cv2.putText(image, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def process_image(detector, spoofer, image_path: str, save_dir: str = 'outputs') -> None:
    """Process a single image for face anti-spoofing detection."""
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
        result = spoofer.predict(image, face.bbox)
        label = 'Real' if result.is_real else 'Fake'
        print(f'  Face {i}: {label} ({result.confidence:.1%})')

        draw_spoofing_result(image, face.bbox, result.is_real, result.confidence)

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{Path(image_path).stem}_spoofing.jpg')
    cv2.imwrite(output_path, image)
    print(f'Output saved: {output_path}')


def process_video(detector, spoofer, video_path: str, save_dir: str = 'outputs') -> None:
    """Process a video file for face anti-spoofing detection."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{Path(video_path).stem}_spoofing.mp4')
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
            result = spoofer.predict(frame, face.bbox)
            draw_spoofing_result(frame, face.bbox, result.is_real, result.confidence)

        out.write(frame)

        if frame_count % 100 == 0:
            print(f'  Processed {frame_count}/{total_frames} frames...')

    cap.release()
    out.release()
    print(f'Done! Output saved: {output_path}')


def run_camera(detector, spoofer, camera_id: int = 0) -> None:
    """Run real-time anti-spoofing detection on webcam."""
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
            result = spoofer.predict(frame, face.bbox)
            draw_spoofing_result(frame, face.bbox, result.is_real, result.confidence)

        cv2.imshow('Face Anti-Spoofing', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Face Anti-Spoofing Detection')
    parser.add_argument('--source', type=str, required=True, help='Image/video path or camera ID (0, 1, ...)')
    parser.add_argument(
        '--model',
        type=str,
        default='v2',
        choices=['v1se', 'v2'],
        help='Model variant: v1se or v2 (default: v2)',
    )
    parser.add_argument('--scale', type=float, default=None, help='Custom crop scale (default: auto)')
    parser.add_argument('--save-dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()

    # Select model variant
    model_name = MiniFASNetWeights.V1SE if args.model == 'v1se' else MiniFASNetWeights.V2

    # Initialize models
    print(f'Initializing models (MiniFASNet {args.model.upper()})...')
    detector = RetinaFace()
    spoofer = create_spoofer(model_name=model_name, scale=args.scale)

    source_type = get_source_type(args.source)

    if source_type == 'camera':
        run_camera(detector, spoofer, int(args.source))
    elif source_type == 'image':
        if not os.path.exists(args.source):
            print(f'Error: Image not found: {args.source}')
            return
        process_image(detector, spoofer, args.source, args.save_dir)
    elif source_type == 'video':
        if not os.path.exists(args.source):
            print(f'Error: Video not found: {args.source}')
            return
        process_video(detector, spoofer, args.source, args.save_dir)
    else:
        print(f"Error: Unknown source type for '{args.source}'")
        print('Supported formats: images (.jpg, .png, ...), videos (.mp4, .avi, ...), or camera ID (0, 1, ...)')


if __name__ == '__main__':
    main()
