# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Face analysis using FaceAnalyzer.

Usage:
    python tools/analyze.py --source path/to/image.jpg
    python tools/analyze.py --source path/to/video.mp4
    python tools/analyze.py --source 0  # webcam
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from _common import get_source_type
from _draw import Card, draw_attribute_cards
import cv2
import numpy as np

from uniface.analyzer import FaceAnalyzer
from uniface.attribute import AgeGender
from uniface.detection import RetinaFace
from uniface.recognition import ArcFace


def face_card(face) -> Card | None:
    """Card content from a FaceAnalyzer result (emotion and/or age + gender)."""
    demo = f'{face.sex}, Age {face.age}' if face.age is not None and face.sex is not None else None
    if face.emotion is not None:
        return Card('Emotion', face.emotion, confidence=face.emotion_confidence, subtitle=demo)
    if demo is not None:
        return Card('Age & Gender', face.sex, subtitle=f'Age {face.age}')
    return None


def process_image(analyzer, image_path: str, save_dir: str = 'outputs', show_similarity: bool = True):
    """Process a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from '{image_path}'")
        return

    faces = analyzer.analyze(image)
    print(f'Detected {len(faces)} face(s)')

    if not faces:
        return

    for i, face in enumerate(faces, 1):
        info = f'  Face {i}: {face.sex}, {face.age}y' if face.age and face.sex else f'  Face {i}'
        if face.embedding is not None:
            info += f' (embedding: {face.embedding.shape})'
        print(info)

    if show_similarity and len(faces) >= 2:
        print('\nSimilarity Matrix:')
        n = len(faces)
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    sim_matrix[i][j] = 1.0
                else:
                    sim = faces[i].compute_similarity(faces[j])
                    sim_matrix[i][j] = sim
                    sim_matrix[j][i] = sim

        print('     ', end='')
        for i in range(n):
            print(f'  F{i + 1:2d}  ', end='')
        print('\n     ' + '-' * (7 * n))

        for i in range(n):
            print(f'F{i + 1:2d} | ', end='')
            for j in range(n):
                print(f'{sim_matrix[i][j]:6.3f} ', end='')
            print()

        pairs = [(i, j, sim_matrix[i][j]) for i in range(n) for j in range(i + 1, n)]
        pairs.sort(key=lambda x: x[2], reverse=True)

        print('\nTop matches (>0.4 = same person):')
        for i, j, sim in pairs[:3]:
            status = 'Same' if sim > 0.4 else 'Different'
            print(f'  Face {i + 1} ↔ Face {j + 1}: {sim:.3f} ({status})')

    cards = []

    for face in faces:
        card = face_card(face)
        if card is not None:
            cards.append((face.bbox, card))
    draw_attribute_cards(image, cards)

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{Path(image_path).stem}_analysis.jpg')
    cv2.imwrite(output_path, image)
    print(f'Output saved: {output_path}')


def process_video(analyzer, video_path: str, save_dir: str = 'outputs'):
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
    output_path = os.path.join(save_dir, f'{Path(video_path).stem}_analysis.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f'Processing video: {video_path} ({total_frames} frames)')
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        faces = analyzer.analyze(frame)

        cards = []

        for face in faces:
            card = face_card(face)
            if card is not None:
                cards.append((face.bbox, card))
        draw_attribute_cards(frame, cards)

        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)

        if frame_count % 100 == 0:
            print(f'  Processed {frame_count}/{total_frames} frames...')

    cap.release()
    out.release()
    print(f'Done! Output saved: {output_path}')


def run_camera(analyzer, camera_id: int = 0):
    """Run real-time analysis on webcam."""
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

        faces = analyzer.analyze(frame)

        cards = []

        for face in faces:
            card = face_card(face)
            if card is not None:
                cards.append((face.bbox, card))
        draw_attribute_cards(frame, cards)

        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Analyzer', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Face analysis with detection, recognition, and attributes')
    parser.add_argument('--source', type=str, required=True, help='Image/video path or camera ID (0, 1, ...)')
    parser.add_argument('--save-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--no-similarity', action='store_true', help='Skip similarity matrix computation')
    args = parser.parse_args()

    detector = RetinaFace()
    recognizer = ArcFace()
    age_gender = AgeGender()
    analyzer = FaceAnalyzer(detector=detector, recognizer=recognizer, predictors=[age_gender])

    source_type = get_source_type(args.source)

    if source_type == 'camera':
        run_camera(analyzer, int(args.source))
    elif source_type == 'image':
        if not os.path.exists(args.source):
            print(f'Error: Image not found: {args.source}')
            return
        process_image(analyzer, args.source, args.save_dir, show_similarity=not args.no_similarity)
    elif source_type == 'video':
        if not os.path.exists(args.source):
            print(f'Error: Video not found: {args.source}')
            return
        process_video(analyzer, args.source, args.save_dir)
    else:
        print(f"Error: Unknown source type for '{args.source}'")
        print('Supported formats: images (.jpg, .png, ...), videos (.mp4, .avi, ...), or camera ID (0, 1, ...)')


if __name__ == '__main__':
    main()
