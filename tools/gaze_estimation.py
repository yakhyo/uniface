# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Gaze estimation on detected faces.

Usage:
    python tools/gaze_estimation.py --source path/to/image.jpg
    python tools/gaze_estimation.py --source path/to/video.mp4
    python tools/gaze_estimation.py --source 0  # webcam
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from uniface import RetinaFace
from uniface.gaze import MobileGaze
from uniface.visualization import draw_gaze

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


def process_image(detector, gaze_estimator, image_path: str, save_dir: str = 'outputs'):
    """Process a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from '{image_path}'")
        return

    faces = detector.detect(image)
    print(f'Detected {len(faces)} face(s)')

    for i, face in enumerate(faces):
        bbox = face.bbox
        x1, y1, x2, y2 = map(int, bbox[:4])
        face_crop = image[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        pitch, yaw = gaze_estimator.estimate(face_crop)
        print(f'  Face {i + 1}: pitch={np.degrees(pitch):.1f}°, yaw={np.degrees(yaw):.1f}°')

        draw_gaze(image, bbox, pitch, yaw, draw_angles=True)

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{Path(image_path).stem}_gaze.jpg')
    cv2.imwrite(output_path, image)
    print(f'Output saved: {output_path}')


def process_video(detector, gaze_estimator, video_path: str, save_dir: str = 'outputs'):
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
    output_path = os.path.join(save_dir, f'{Path(video_path).stem}_gaze.mp4')
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
            bbox = face.bbox
            x1, y1, x2, y2 = map(int, bbox[:4])
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            pitch, yaw = gaze_estimator.estimate(face_crop)
            draw_gaze(frame, bbox, pitch, yaw)

        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)

        if frame_count % 100 == 0:
            print(f'  Processed {frame_count}/{total_frames} frames...')

    cap.release()
    out.release()
    print(f'Done! Output saved: {output_path}')


def run_camera(detector, gaze_estimator, camera_id: int = 0):
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
            bbox = face.bbox
            x1, y1, x2, y2 = map(int, bbox[:4])
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            pitch, yaw = gaze_estimator.estimate(face_crop)
            draw_gaze(frame, bbox, pitch, yaw)

        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Gaze Estimation', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Run gaze estimation')
    parser.add_argument('--source', type=str, required=True, help='Image/video path or camera ID (0, 1, ...)')
    parser.add_argument('--save-dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()

    detector = RetinaFace()
    gaze_estimator = MobileGaze()

    source_type = get_source_type(args.source)

    if source_type == 'camera':
        run_camera(detector, gaze_estimator, int(args.source))
    elif source_type == 'image':
        if not os.path.exists(args.source):
            print(f'Error: Image not found: {args.source}')
            return
        process_image(detector, gaze_estimator, args.source, args.save_dir)
    elif source_type == 'video':
        if not os.path.exists(args.source):
            print(f'Error: Video not found: {args.source}')
            return
        process_video(detector, gaze_estimator, args.source, args.save_dir)
    else:
        print(f"Error: Unknown source type for '{args.source}'")
        print('Supported formats: images (.jpg, .png, ...), videos (.mp4, .avi, ...), or camera ID (0, 1, ...)')


if __name__ == '__main__':
    main()
