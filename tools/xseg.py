# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""XSeg face segmentation on detected faces.

Usage:
    python tools/xseg.py --source path/to/image.jpg
    python tools/xseg.py --source path/to/video.mp4
    python tools/xseg.py --source 0  # webcam
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from _common import get_source_type
import cv2
import numpy as np

from uniface.detection import RetinaFace
from uniface.parsing import XSeg


def apply_mask_visualization(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Apply colored mask overlay for visualization."""
    overlay = image.copy().astype(np.float32)
    mask_3ch = np.stack([mask * 0.3, mask * 0.7, mask * 0.3], axis=-1)
    overlay = overlay * (1 - mask[..., None] * alpha) + mask_3ch * 255 * alpha

    return overlay.clip(0, 255).astype(np.uint8)


def process_image(
    detector: RetinaFace,
    parser: XSeg,
    image_path: str,
    save_dir: str = 'outputs',
) -> None:
    """Process a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from '{image_path}'")
        return

    faces = detector.detect(image)
    print(f'Detected {len(faces)} face(s)')

    if len(faces) == 0:
        print('No faces detected.')
        return

    # Accumulate masks from all faces
    full_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    for i, face in enumerate(faces):
        if face.landmarks is None:
            print(f'  Face {i + 1}: skipped (no landmarks)')
            continue

        mask = parser.parse(image, face.landmarks)
        full_mask = np.maximum(full_mask, mask)
        print(f'  Face {i + 1}: done')

    # Apply visualization
    result_image = apply_mask_visualization(image, full_mask)

    # Draw bounding boxes
    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox[:4])
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{Path(image_path).stem}_xseg.jpg')
    cv2.imwrite(output_path, result_image)
    print(f'Output saved: {output_path}')

    mask_path = os.path.join(save_dir, f'{Path(image_path).stem}_xseg_mask.png')
    mask_uint8 = (full_mask * 255).astype(np.uint8)
    cv2.imwrite(mask_path, mask_uint8)
    print(f'Mask saved: {mask_path}')


def process_video(
    detector: RetinaFace,
    parser: XSeg,
    video_path: str,
    save_dir: str = 'outputs',
) -> None:
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
    output_path = os.path.join(save_dir, f'{Path(video_path).stem}_xseg.mp4')
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

        # Accumulate masks from all faces
        full_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        for face in faces:
            if face.landmarks is None:
                continue
            mask = parser.parse(frame, face.landmarks)
            full_mask = np.maximum(full_mask, mask)

        # Apply visualization
        result_frame = apply_mask_visualization(frame, full_mask)

        # Draw bounding boxes
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox[:4])
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(result_frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(result_frame)

        if frame_count % 100 == 0:
            print(f'  Processed {frame_count}/{total_frames} frames...')

    cap.release()
    out.release()
    print(f'Done! Output saved: {output_path}')


def run_camera(
    detector: RetinaFace,
    parser: XSeg,
    camera_id: int = 0,
) -> None:
    """Run real-time detection on webcam."""
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

        # Accumulate masks from all faces
        full_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        for face in faces:
            if face.landmarks is None:
                continue
            mask = parser.parse(frame, face.landmarks)
            full_mask = np.maximum(full_mask, mask)

        # Apply visualization
        result_frame = apply_mask_visualization(frame, full_mask)

        # Draw bounding boxes
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox[:4])
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(result_frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('XSeg Face Segmentation', result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    arg_parser = argparse.ArgumentParser(description='XSeg face segmentation')
    arg_parser.add_argument('--source', type=str, required=True, help='Image/video path or camera ID (0, 1, ...)')
    arg_parser.add_argument('--save-dir', type=str, default='outputs', help='Output directory')
    arg_parser.add_argument(
        '--blur',
        type=float,
        default=0,
        help='Gaussian blur sigma for mask smoothing (default: 0 = raw)',
    )
    arg_parser.add_argument(
        '--align-size',
        type=int,
        default=256,
        help='Face alignment size (default: 256)',
    )
    args = arg_parser.parse_args()

    # Initialize models
    detector = RetinaFace()
    parser = XSeg(blur_sigma=args.blur, align_size=args.align_size)

    source_type = get_source_type(args.source)

    if source_type == 'camera':
        run_camera(detector, parser, int(args.source))
    elif source_type == 'image':
        if not os.path.exists(args.source):
            print(f'Error: Image not found: {args.source}')
            return
        process_image(detector, parser, args.source, args.save_dir)
    elif source_type == 'video':
        if not os.path.exists(args.source):
            print(f'Error: Video not found: {args.source}')
            return
        process_video(detector, parser, args.source, args.save_dir)
    else:
        print(f"Error: Unknown source type for '{args.source}'")
        print('Supported formats: images (.jpg, .png, ...), videos (.mp4, .avi, ...), or camera ID (0, 1, ...)')


if __name__ == '__main__':
    main()
