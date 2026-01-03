# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Face detection on image, video, or webcam.

Usage:
    python tools/detection.py --source path/to/image.jpg
    python tools/detection.py --source path/to/video.mp4
    python tools/detection.py --source 0  # webcam
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2

from uniface.detection import SCRFD, RetinaFace, YOLOv5Face, YOLOv8Face
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


def process_image(detector, image_path: str, threshold: float = 0.6, save_dir: str = 'outputs'):
    """Process a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from '{image_path}'")
        return

    faces = detector.detect(image)

    if faces:
        bboxes = [face.bbox for face in faces]
        scores = [face.confidence for face in faces]
        landmarks = [face.landmarks for face in faces]
        draw_detections(image, bboxes, scores, landmarks, vis_threshold=threshold)

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{os.path.splitext(os.path.basename(image_path))[0]}_out.jpg')
    cv2.imwrite(output_path, image)
    print(f'Detected {len(faces)} face(s). Output saved: {output_path}')


def process_video(detector, video_path: str, threshold: float = 0.6, save_dir: str = 'outputs'):
    """Process a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{Path(video_path).stem}_out.mp4')
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
            image=frame,
            bboxes=bboxes,
            scores=scores,
            landmarks=landmarks,
            vis_threshold=threshold,
            draw_score=True,
            fancy_bbox=True,
        )

        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)

        # Show progress
        if frame_count % 100 == 0:
            print(f'  Processed {frame_count}/{total_frames} frames...')

    cap.release()
    out.release()
    print(f'Done! Output saved: {output_path}')


def run_camera(detector, camera_id: int = 0, threshold: float = 0.6):
    """Run real-time detection on webcam."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f'Cannot open camera {camera_id}')
        return

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        if not ret:
            break

        faces = detector.detect(frame)

        bboxes = [f.bbox for f in faces]
        scores = [f.confidence for f in faces]
        landmarks = [f.landmarks for f in faces]
        draw_detections(
            image=frame,
            bboxes=bboxes,
            scores=scores,
            landmarks=landmarks,
            vis_threshold=threshold,
            draw_score=True,
            fancy_bbox=True,
        )

        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Run face detection')
    parser.add_argument('--source', type=str, required=True, help='Image/video path or camera ID (0, 1, ...)')
    parser.add_argument(
        '--method', type=str, default='retinaface', choices=['retinaface', 'scrfd', 'yolov5face', 'yolov8face']
    )
    parser.add_argument('--threshold', type=float, default=0.25, help='Visualization threshold')
    parser.add_argument('--save-dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()

    # Initialize detector
    if args.method == 'retinaface':
        detector = RetinaFace()
    elif args.method == 'scrfd':
        detector = SCRFD()
    elif args.method == 'yolov5face':
        from uniface.constants import YOLOv5FaceWeights

        detector = YOLOv5Face(model_name=YOLOv5FaceWeights.YOLOV5M)
    else:  # yolov8face
        from uniface.constants import YOLOv8FaceWeights

        detector = YOLOv8Face(model_name=YOLOv8FaceWeights.YOLOV8N)

    # Determine source type and process
    source_type = get_source_type(args.source)

    if source_type == 'camera':
        run_camera(detector, int(args.source), args.threshold)
    elif source_type == 'image':
        if not os.path.exists(args.source):
            print(f'Error: Image not found: {args.source}')
            return
        process_image(detector, args.source, args.threshold, args.save_dir)
    elif source_type == 'video':
        if not os.path.exists(args.source):
            print(f'Error: Video not found: {args.source}')
            return
        process_video(detector, args.source, args.threshold, args.save_dir)
    else:
        print(f"Error: Unknown source type for '{args.source}'")
        print('Supported formats: images (.jpg, .png, ...), videos (.mp4, .avi, ...), or camera ID (0, 1, ...)')


if __name__ == '__main__':
    main()
