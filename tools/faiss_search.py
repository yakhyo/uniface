# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""FAISS index build and multi-identity face search.

Build a vector index from a directory of person sub-folders, then search
against it in a video or webcam stream.

Usage:
    python tools/faiss_search.py build --faces-dir dataset/ --db-path ./vector_index
    python tools/faiss_search.py run   --db-path ./vector_index --source video.mp4
    python tools/faiss_search.py run   --db-path ./vector_index --source 0  # webcam
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from _common import IMAGE_EXTENSIONS, get_source_type
import cv2

from uniface import create_detector, create_recognizer
from uniface.draw import draw_corner_bbox, draw_text_label
from uniface.indexing import FAISS


def _draw_face(image, bbox, text: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = map(int, bbox[:4])
    thickness = max(round(sum(image.shape[:2]) / 2 * 0.003), 2)
    font_scale = max(0.4, min(0.7, (y2 - y1) / 200))
    draw_corner_bbox(image, (x1, y1, x2, y2), color=color, thickness=thickness)
    draw_text_label(image, text, x1, y1, bg_color=color, font_scale=font_scale)


def process_frame(frame, detector, recognizer, store: FAISS, threshold: float = 0.4):
    faces = detector.detect(frame)
    if not faces:
        return frame

    for face in faces:
        embedding = recognizer.get_normalized_embedding(frame, face.landmarks)
        result, sim = store.search(embedding, threshold=threshold)

        text = f'{result["person_id"]} ({sim:.2f})' if result else f'Unknown ({sim:.2f})'
        color = (0, 255, 0) if result else (0, 0, 255)
        _draw_face(frame, face.bbox, text, color)

    return frame


def process_video(detector, recognizer, store: FAISS, video_path: str, save_dir: str, threshold: float = 0.4):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{Path(video_path).stem}_faiss_search.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f'Processing video: {video_path} ({total_frames} frames)')
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = process_frame(frame, detector, recognizer, store, threshold)
        out.write(frame)

        if frame_count % 100 == 0:
            print(f'  Processed {frame_count}/{total_frames} frames...')

    cap.release()
    out.release()
    print(f'Done! Output saved: {output_path}')


def run_camera(detector, recognizer, store: FAISS, camera_id: int = 0, threshold: float = 0.4):
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

        frame = process_frame(frame, detector, recognizer, store, threshold)

        cv2.imshow('Vector Search', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def build(args: argparse.Namespace) -> None:
    faces_dir = Path(args.faces_dir)
    if not faces_dir.is_dir():
        print(f"Error: '{faces_dir}' is not a directory")
        return

    detector = create_detector()
    recognizer = create_recognizer()
    store = FAISS(db_path=args.db_path)

    persons = sorted(p.name for p in faces_dir.iterdir() if p.is_dir())
    if not persons:
        print(f"Error: No sub-folders found in '{faces_dir}'")
        return

    print(f'Found {len(persons)} persons: {", ".join(persons)}')

    total_added = 0
    for person_id in persons:
        person_dir = faces_dir / person_id
        images = [f for f in person_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]

        added = 0
        for img_path in images:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f'  Warning: Failed to read {img_path}, skipping')
                continue

            faces = detector.detect(image)
            if not faces:
                print(f'  Warning: No face detected in {img_path}, skipping')
                continue

            embedding = recognizer.get_normalized_embedding(image, faces[0].landmarks)
            store.add(embedding, {'person_id': person_id, 'source': str(img_path)})
            added += 1

        total_added += added
        if added:
            print(f'  {person_id}: {added} embeddings added')
        else:
            print(f'  {person_id}: no valid faces found')

    store.save()
    print(f'\nIndex saved to {args.db_path} ({total_added} vectors, {len(persons)} persons)')


def run(args: argparse.Namespace) -> None:
    detector = create_detector()
    recognizer = create_recognizer()

    store = FAISS(db_path=args.db_path)
    if not store.load():
        print(f"Error: No index found at '{args.db_path}'")
        return
    print(f'Loaded FAISS index: {store}')

    source_type = get_source_type(args.source)

    if source_type == 'camera':
        run_camera(detector, recognizer, store, int(args.source), args.threshold)
    elif source_type == 'video':
        if not os.path.exists(args.source):
            print(f'Error: Video not found: {args.source}')
            return
        process_video(detector, recognizer, store, args.source, args.save_dir, args.threshold)
    else:
        print(f"Error: Source must be a video file or camera ID, not '{args.source}'")


def main():
    parser = argparse.ArgumentParser(description='FAISS vector search')
    sub = parser.add_subparsers(dest='command', required=True)

    build_p = sub.add_parser('build', help='Build a FAISS index from person sub-folders')
    build_p.add_argument('--faces-dir', type=str, required=True, help='Directory with person sub-folders')
    build_p.add_argument('--db-path', type=str, default='./vector_index', help='Where to save the index')

    run_p = sub.add_parser('run', help='Search faces against a FAISS index')
    run_p.add_argument('--db-path', type=str, required=True, help='Path to saved FAISS index')
    run_p.add_argument('--source', type=str, required=True, help='Video path or camera ID')
    run_p.add_argument('--threshold', type=float, default=0.4, help='Similarity threshold')
    run_p.add_argument('--save-dir', type=str, default='outputs', help='Output directory')

    args = parser.parse_args()

    if args.command == 'build':
        build(args)
    elif args.command == 'run':
        run(args)


if __name__ == '__main__':
    main()
