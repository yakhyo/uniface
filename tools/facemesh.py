# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Dense 3D face mesh (MediaPipe): 468 landmarks, or 478 with irises.

Usage:
    python tools/facemesh.py --source path/to/image.jpg
    python tools/facemesh.py --source path/to/video.mp4 --mode points
    python tools/facemesh.py --source 0  # webcam
    python tools/facemesh.py --source image.jpg --detector blazeface  # MediaPipe parity
    python tools/facemesh.py --source 0 --model v2_478  # with iris landmarks
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from _common import get_source_type
import cv2

from uniface.constants import FaceMeshWeights
from uniface.detection import SCRFD, BlazeFace, RetinaFace
from uniface.draw import draw_mesh
from uniface.landmark import FaceMesh

DETECTORS = {'scrfd': SCRFD, 'retinaface': RetinaFace, 'blazeface': BlazeFace}
MODELS = {w.name.lower(): w for w in FaceMeshWeights}


def annotate(image, detector, mesher, mode: str) -> int:
    """Detect, mesh, and draw every face. Returns the number of faces found."""
    faces = detector.detect(image)
    if not faces:
        return 0

    # One batched inference for every face in the frame.
    for result in mesher.predict(image, faces):
        draw_mesh(image, result.landmarks, mode=mode)

    return len(faces)


def process_image(detector, mesher, image_path: str, mode: str, save_dir: str = 'outputs'):
    """Process a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from '{image_path}'")
        return

    faces = detector.detect(image)
    print(f'Detected {len(faces)} face(s)')

    if faces:
        results = mesher.predict(image, faces)
        for i, result in enumerate(results):
            print(f'  Face {i + 1}: {result.landmarks.shape[0]} landmarks, presence {result.score:.3f}')
            draw_mesh(image, result.landmarks, mode=mode)

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{Path(image_path).stem}_facemesh.jpg')
    cv2.imwrite(output_path, image)
    print(f'Output saved: {output_path}')


def process_video(detector, mesher, video_path: str, mode: str, save_dir: str = 'outputs'):
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
    output_path = os.path.join(save_dir, f'{Path(video_path).stem}_facemesh.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f'Processing video: {video_path} ({total_frames} frames)')
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        num_faces = annotate(frame, detector, mesher, mode)

        cv2.putText(frame, f'Faces: {num_faces}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)

        if frame_count % 100 == 0:
            print(f'  Processed {frame_count}/{total_frames} frames...')

    cap.release()
    out.release()
    print(f'Done! Output saved: {output_path}')


def run_camera(detector, mesher, mode: str, camera_id: int = 0):
    """Run real-time face mesh on webcam."""
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

        num_faces = annotate(frame, detector, mesher, mode)

        cv2.putText(frame, f'Faces: {num_faces}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(f'Face Mesh ({mesher.num_landmarks} points)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Run dense 3D face mesh (468 or 478 landmarks)')
    parser.add_argument('--source', type=str, required=True, help='Image/video path or camera ID (0, 1, ...)')
    parser.add_argument(
        '--detector',
        type=str,
        default='scrfd',
        choices=list(DETECTORS),
        help="Seed detector. 'blazeface' reproduces MediaPipe's own pipeline.",
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='partial',
        choices=['full', 'partial', 'points'],
        help="Render style. 'full' is the dense tessellation and is slow for video.",
    )
    parser.add_argument(
        '--model',
        type=str,
        default='v1_468',
        choices=list(MODELS),
        help="'v1_468' is the classic mesh; 'v2_478' adds iris landmarks at ~3x the compute.",
    )
    parser.add_argument('--save-dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()

    detector = DETECTORS[args.detector]()
    mesher = FaceMesh(model_name=MODELS[args.model])

    source_type = get_source_type(args.source)

    if source_type == 'camera':
        run_camera(detector, mesher, args.mode, int(args.source))
    elif source_type == 'video':
        process_video(detector, mesher, args.source, args.mode, args.save_dir)
    else:
        process_image(detector, mesher, args.source, args.mode, args.save_dir)


if __name__ == '__main__':
    main()
