# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Batch face detection on a folder of images.

Usage:
    python tools/batch_process.py --input images/ --output results/
"""

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from uniface.detection import SCRFD, RetinaFace
from uniface.draw import draw_detections


def get_image_files(input_dir: Path, extensions: tuple) -> list:
    files = []
    for ext in extensions:
        files.extend(input_dir.glob(f'*.{ext}'))
        files.extend(input_dir.glob(f'*.{ext.upper()}'))
    return sorted(files)


def process_image(detector, image_path: Path, output_path: Path, threshold: float) -> int:
    """Process single image. Returns face count or -1 on error."""
    image = cv2.imread(str(image_path))
    if image is None:
        return -1

    faces = detector.detect(image)

    # unpack face data for visualization
    bboxes = [f.bbox for f in faces]
    scores = [f.confidence for f in faces]
    landmarks = [f.landmarks for f in faces]
    draw_detections(
        image=image, bboxes=bboxes, scores=scores, landmarks=landmarks, vis_threshold=threshold, corner_bbox=True
    )

    cv2.putText(
        image,
        f'Faces: {len(faces)}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.imwrite(str(output_path), image)

    return len(faces)


def main():
    parser = argparse.ArgumentParser(description='Batch process images with face detection')
    parser.add_argument('--input', type=str, required=True, help='Input directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--detector', type=str, default='retinaface', choices=['retinaface', 'scrfd'])
    parser.add_argument('--threshold', type=float, default=0.6, help='Visualization threshold')
    parser.add_argument('--extensions', type=str, default='jpg,jpeg,png,bmp', help='Image extensions')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input directory '{args.input}' does not exist")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    extensions = tuple(ext.strip() for ext in args.extensions.split(','))
    image_files = get_image_files(input_path, extensions)

    if not image_files:
        print(f'No images found with extensions {extensions}')
        return

    print(f'Found {len(image_files)} images')

    detector = RetinaFace() if args.detector == 'retinaface' else SCRFD()

    success, errors, total_faces = 0, 0, 0

    for img_path in tqdm(image_files, desc='Processing', unit='img'):
        out_path = output_path / f'{img_path.stem}_detected{img_path.suffix}'
        result = process_image(detector, img_path, out_path, args.threshold)

        if result >= 0:
            success += 1
            total_faces += result
        else:
            errors += 1
            print(f'\nFailed: {img_path.name}')

    print(f'\nDone! {success} processed, {errors} errors, {total_faces} faces total')


if __name__ == '__main__':
    main()
