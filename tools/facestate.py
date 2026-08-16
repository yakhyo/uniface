# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Face state prediction on detected faces with FaceAttribNet.

Predicts five independent binary attributes per face: left/right eye openness,
eyeglasses, face mask, and sunglasses. Each probability comes from its own
classifier head, so several can be high at once; each is thresholded separately.

Usage:
    python tools/facestate.py --source path/to/image.jpg
    python tools/facestate.py --source path/to/video.mp4
    python tools/facestate.py --source 0  # webcam
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from _common import get_source_type
import cv2

from uniface.attribute import FaceAttribNet
from uniface.detection import SCRFD, RetinaFace
from uniface.types import FaceStateResult

INK, GREEN, RED = (55, 41, 31), (45, 128, 21), (28, 28, 185)  # BGR


def draw_face_states(image, bbox, result: FaceStateResult, threshold: float = 0.5) -> None:
    """Draw a face bounding box and a True/False panel for every attribute.

    The panel is placed above the box when there is room, otherwise inside it.
    Panel size scales with the box width so it stays readable on both small
    and large faces.

    Args:
        image: Image to annotate in-place (BGR).
        bbox: Face bounding box [x1, y1, x2, y2].
        result: Predicted per-attribute probabilities.
        threshold: Probability above which an attribute counts as present.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox[:4])
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    probs = result.as_dict()

    def tf(prob: float) -> tuple[str, tuple[int, int, int]]:
        return ('True', GREEN) if prob > threshold else ('False', RED)

    rows = [
        [('Eyes ', INK), tf(probs['left_eye_open']), (', ', INK), tf(probs['right_eye_open'])],
        [('Mask ', INK), tf(probs['mask'])],
        [('Glasses ', INK), tf(probs['eyeglasses'])],
        [('Sunglasses ', INK), tf(probs['sunglasses'])],
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(1.6, (x2 - x1) / 350))
    thickness = 1 if scale < 0.7 else 2
    line_h, pad = int(38 * scale), int(8 * scale)

    panel_w = max(sum(cv2.getTextSize(t, font, scale, thickness)[0][0] for t, _ in row) for row in rows)
    panel_h = len(rows) * line_h + pad
    y_top = y1 - panel_h - pad if y1 - panel_h - pad >= 0 else y1 + pad  # above the box, else inside it
    cv2.rectangle(image, (x1, y_top), (x1 + panel_w + 2 * pad, y_top + panel_h), (255, 255, 255), -1)
    for i, row in enumerate(rows):
        x, y = x1 + pad, y_top + (i + 1) * line_h - int(6 * scale)
        for text, color in row:
            cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
            x += cv2.getTextSize(text, font, scale, thickness)[0][0]


def process_image(
    detector,
    face_attrib,
    image_path: str,
    save_dir: str = 'outputs',
    threshold: float = 0.5,
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

    for i, face in enumerate(faces):
        result = face_attrib.predict(image, face)
        active = ', '.join(result.labels(threshold)) or 'none'
        print(f'  Face {i + 1}: {result}')
        print(f'  Face {i + 1}: active -> {active}')
        draw_face_states(image, face.bbox, result, threshold)

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{Path(image_path).stem}_facestate.jpg')
    cv2.imwrite(output_path, image)
    print(f'Output saved: {output_path}')


def process_video(
    detector,
    face_attrib,
    video_path: str,
    save_dir: str = 'outputs',
    threshold: float = 0.5,
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
    output_path = os.path.join(save_dir, f'{Path(video_path).stem}_facestate.mp4')
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
            result = face_attrib.predict(frame, face)
            draw_face_states(frame, face.bbox, result, threshold)

        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)

        if frame_count % 100 == 0:
            print(f'  Processed {frame_count}/{total_frames} frames...')

    cap.release()
    out.release()
    print(f'Done! Output saved: {output_path}')


def run_camera(detector, face_attrib, camera_id: int = 0, threshold: float = 0.5):
    """Run real-time face state prediction on webcam."""
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
            result = face_attrib.predict(frame, face)
            draw_face_states(frame, face.bbox, result, threshold)

        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face State Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Run face state prediction (FaceAttribNet)')
    parser.add_argument('--source', type=str, required=True, help='Image/video path or camera ID (0, 1, ...)')
    parser.add_argument('--detector', type=str, default='retinaface', choices=['retinaface', 'scrfd'])
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability above which an attribute is present')
    parser.add_argument('--margin', type=float, default=0.0, help='Fraction to expand each face crop by on each side')
    parser.add_argument('--save-dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()

    detector = RetinaFace() if args.detector == 'retinaface' else SCRFD()
    face_attrib = FaceAttribNet(margin=args.margin)

    source_type = get_source_type(args.source)

    if source_type == 'camera':
        run_camera(detector, face_attrib, int(args.source), args.threshold)
    elif source_type == 'image':
        if not os.path.exists(args.source):
            print(f'Error: Image not found: {args.source}')
            return
        process_image(detector, face_attrib, args.source, args.save_dir, args.threshold)
    elif source_type == 'video':
        if not os.path.exists(args.source):
            print(f'Error: Video not found: {args.source}')
            return
        process_video(detector, face_attrib, args.source, args.save_dir, args.threshold)
    else:
        print(f"Error: Unknown source type for '{args.source}'")
        print('Supported formats: images (.jpg, .png, ...), videos (.mp4, .avi, ...), or camera ID (0, 1, ...)')


if __name__ == '__main__':
    main()
