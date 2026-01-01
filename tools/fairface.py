# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""FairFace attribute prediction (race, gender, age) on detected faces.

Usage:
    python tools/fairface.py --source path/to/image.jpg
    python tools/fairface.py --source path/to/video.mp4
    python tools/fairface.py --source 0  # webcam
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2

from uniface import SCRFD, RetinaFace
from uniface.attribute import FairFace
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


def draw_fairface_label(image, bbox, sex: str, age_group: str, race: str):
    """Draw FairFace attributes above the bounding box."""
    x1, y1 = int(bbox[0]), int(bbox[1])
    text = f'{sex}, {age_group}, {race}'
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 255, 0), -1)
    cv2.putText(image, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


def process_image(
    detector,
    fairface,
    image_path: str,
    save_dir: str = 'outputs',
    threshold: float = 0.6,
):
    """Process a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from '{image_path}'")
        return

    faces = detector.detect(image)
    print(f'Detected {len(faces)} face(s)')

    if not faces:
        return

    bboxes = [f.bbox for f in faces]
    scores = [f.confidence for f in faces]
    landmarks = [f.landmarks for f in faces]
    draw_detections(
        image=image, bboxes=bboxes, scores=scores, landmarks=landmarks, vis_threshold=threshold, fancy_bbox=True
    )

    for i, face in enumerate(faces):
        result = fairface.predict(image, face.bbox)
        print(f'  Face {i + 1}: {result.sex}, {result.age_group}, {result.race}')
        draw_fairface_label(image, face.bbox, result.sex, result.age_group, result.race)

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{Path(image_path).stem}_fairface.jpg')
    cv2.imwrite(output_path, image)
    print(f'Output saved: {output_path}')


def process_video(
    detector,
    fairface,
    video_path: str,
    save_dir: str = 'outputs',
    threshold: float = 0.6,
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
    output_path = os.path.join(save_dir, f'{Path(video_path).stem}_fairface.mp4')
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

        bboxes = [f.bbox for f in faces]
        scores = [f.confidence for f in faces]
        landmarks = [f.landmarks for f in faces]
        draw_detections(
            image=frame, bboxes=bboxes, scores=scores, landmarks=landmarks, vis_threshold=threshold, fancy_bbox=True
        )

        for face in faces:
            result = fairface.predict(frame, face.bbox)
            draw_fairface_label(frame, face.bbox, result.sex, result.age_group, result.race)

        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)

        if frame_count % 100 == 0:
            print(f'  Processed {frame_count}/{total_frames} frames...')

    cap.release()
    out.release()
    print(f'Done! Output saved: {output_path}')


def run_camera(detector, fairface, camera_id: int = 0, threshold: float = 0.6):
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

        for face in faces:
            result = fairface.predict(frame, face.bbox)
            draw_fairface_label(frame, face.bbox, result.sex, result.age_group, result.race)

        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('FairFace Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Run FairFace attribute prediction (race, gender, age)')
    parser.add_argument('--source', type=str, required=True, help='Image/video path or camera ID (0, 1, ...)')
    parser.add_argument('--detector', type=str, default='retinaface', choices=['retinaface', 'scrfd'])
    parser.add_argument('--threshold', type=float, default=0.6, help='Visualization threshold')
    parser.add_argument('--save-dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()

    detector = RetinaFace() if args.detector == 'retinaface' else SCRFD()
    fairface = FairFace()

    source_type = get_source_type(args.source)

    if source_type == 'camera':
        run_camera(detector, fairface, int(args.source), args.threshold)
    elif source_type == 'image':
        if not os.path.exists(args.source):
            print(f'Error: Image not found: {args.source}')
            return
        process_image(detector, fairface, args.source, args.save_dir, args.threshold)
    elif source_type == 'video':
        if not os.path.exists(args.source):
            print(f'Error: Video not found: {args.source}')
            return
        process_video(detector, fairface, args.source, args.save_dir, args.threshold)
    else:
        print(f"Error: Unknown source type for '{args.source}'")
        print('Supported formats: images (.jpg, .png, ...), videos (.mp4, .avi, ...), or camera ID (0, 1, ...)')


if __name__ == '__main__':
    main()
