# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Face parsing on detected faces.

Usage:
    python tools/parse.py --source path/to/image.jpg
    python tools/parse.py --source path/to/video.mp4
    python tools/parse.py --source 0  # webcam
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from _common import get_source_type
import cv2
import numpy as np

from uniface.constants import ParsingWeights
from uniface.detection import RetinaFace
from uniface.draw import vis_parsing_maps
from uniface.parsing import BiSeNet


def expand_bbox(
    bbox: np.ndarray,
    image_shape: tuple[int, int],
    expand_ratio: float = 0.2,
    expand_top_ratio: float = 0.4,
) -> tuple[int, int, int, int]:
    """
    Expand bounding box to include full head region for face parsing.

    Face detection typically returns tight face boxes, but face parsing
    requires the full head including hair, ears, and neck.

    Args:
        bbox: Original bounding box [x1, y1, x2, y2].
        image_shape: Image dimensions as (height, width).
        expand_ratio: Expansion ratio for left, right, and bottom (default: 0.2 = 20%).
        expand_top_ratio: Expansion ratio for top to capture hair/forehead (default: 0.4 = 40%).

    Returns:
        Tuple[int, int, int, int]: Expanded bbox (x1, y1, x2, y2) clamped to image bounds.
    """
    x1, y1, x2, y2 = map(int, bbox[:4])
    height, width = image_shape[:2]

    face_width = x2 - x1
    face_height = y2 - y1

    expand_x = int(face_width * expand_ratio)
    expand_y_bottom = int(face_height * expand_ratio)
    expand_y_top = int(face_height * expand_top_ratio)

    new_x1 = max(0, x1 - expand_x)
    new_y1 = max(0, y1 - expand_y_top)
    new_x2 = min(width, x2 + expand_x)
    new_y2 = min(height, y2 + expand_y_bottom)

    return new_x1, new_y1, new_x2, new_y2


def process_image(detector, parser, image_path: str, save_dir: str = 'outputs', expand_ratio: float = 0.2):
    """Process a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from '{image_path}'")
        return

    faces = detector.detect(image)
    print(f'Detected {len(faces)} face(s)')

    result_image = image.copy()

    for i, face in enumerate(faces):
        x1, y1, x2, y2 = expand_bbox(face.bbox, image.shape, expand_ratio=expand_ratio)
        face_crop = image[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        mask = parser.parse(face_crop)
        print(f'  Face {i + 1}: parsed with {len(set(mask.flatten()))} unique classes')

        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        vis_result = vis_parsing_maps(face_crop_rgb, mask, save_image=False)

        result_image[y1:y2, x1:x2] = vis_result
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{Path(image_path).stem}_parsing.jpg')
    cv2.imwrite(output_path, result_image)
    print(f'Output saved: {output_path}')


def process_video(detector, parser, video_path: str, save_dir: str = 'outputs', expand_ratio: float = 0.2):
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
    output_path = os.path.join(save_dir, f'{Path(video_path).stem}_parsing.mp4')
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
            x1, y1, x2, y2 = expand_bbox(face.bbox, frame.shape, expand_ratio=expand_ratio)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            mask = parser.parse(face_crop)
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            vis_result = vis_parsing_maps(face_crop_rgb, mask, save_image=False)

            frame[y1:y2, x1:x2] = vis_result
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)

        if frame_count % 100 == 0:
            print(f'  Processed {frame_count}/{total_frames} frames...')

    cap.release()
    out.release()
    print(f'Done! Output saved: {output_path}')


def run_camera(detector, parser, camera_id: int = 0, expand_ratio: float = 0.2):
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

        for face in faces:
            x1, y1, x2, y2 = expand_bbox(face.bbox, frame.shape, expand_ratio=expand_ratio)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            mask = parser.parse(face_crop)
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            vis_result = vis_parsing_maps(face_crop_rgb, mask, save_image=False)

            frame[y1:y2, x1:x2] = vis_result
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Parsing', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser_arg = argparse.ArgumentParser(description='Run face parsing')
    parser_arg.add_argument('--source', type=str, required=True, help='Image/video path or camera ID (0, 1, ...)')
    parser_arg.add_argument('--save-dir', type=str, default='outputs', help='Output directory')
    parser_arg.add_argument(
        '--model', type=str, default=ParsingWeights.RESNET18, choices=[ParsingWeights.RESNET18, ParsingWeights.RESNET34]
    )
    parser_arg.add_argument(
        '--expand-ratio',
        type=float,
        default=0.2,
        help='Bbox expansion ratio for full head coverage (default: 0.2 = 20%%)',
    )
    args = parser_arg.parse_args()

    detector = RetinaFace()
    parser = BiSeNet(model_name=args.model)

    source_type = get_source_type(args.source)

    if source_type == 'camera':
        run_camera(detector, parser, int(args.source), expand_ratio=args.expand_ratio)
    elif source_type == 'image':
        if not os.path.exists(args.source):
            print(f'Error: Image not found: {args.source}')
            return
        process_image(detector, parser, args.source, args.save_dir, expand_ratio=args.expand_ratio)
    elif source_type == 'video':
        if not os.path.exists(args.source):
            print(f'Error: Video not found: {args.source}')
            return
        process_video(detector, parser, args.source, args.save_dir, expand_ratio=args.expand_ratio)
    else:
        print(f"Error: Unknown source type for '{args.source}'")
        print('Supported formats: images (.jpg, .png, ...), videos (.mp4, .avi, ...), or camera ID (0, 1, ...)')


if __name__ == '__main__':
    main()
