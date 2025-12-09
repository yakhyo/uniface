#!/usr/bin/env python3
"""
Face Detection Demo Script

This script demonstrates how to use UniFace for face detection on images.
It supports multiple detection methods (RetinaFace, SCRFD, YOLOv5Face) and
can process single images or directories of images.

Usage:
    python face_detection_demo.py path/to/image.jpg
    python face_detection_demo.py path/to/images/ --method scrfd
    python face_detection_demo.py image.jpg --output results/ --threshold 0.7
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from uniface.detection import SCRFD, RetinaFace, YOLOv5Face
from uniface.visualization import draw_detections


def create_detector(method: str, conf_threshold: float = 0.5):
    """
    Create a face detector based on the specified method.

    Args:
        method: Detection method ('retinaface', 'scrfd', 'yolov5face')
        conf_threshold: Confidence threshold for detections

    Returns:
        Face detector instance
    """
    detectors = {
        'retinaface': lambda: RetinaFace(conf_thresh=conf_threshold),
        'scrfd': lambda: SCRFD(conf_thresh=conf_threshold),
        'yolov5face': lambda: YOLOv5Face(conf_thresh=conf_threshold),
    }

    if method not in detectors:
        raise ValueError(f"Unknown method: {method}. Choose from: {list(detectors.keys())}")

    return detectors[method]()


def detect_faces(detector, image: np.ndarray) -> list[dict]:
    """
    Run face detection on an image.

    Args:
        detector: Face detector instance
        image: Input image (BGR format from cv2)

    Returns:
        List of detection dictionaries with keys:
        - 'bbox': [x1, y1, x2, y2] bounding box coordinates
        - 'confidence': Detection confidence score (0-1)
        - 'landmarks': 5-point facial landmarks array (5, 2)
    """
    return detector.detect(image)


def process_image(
    detector,
    image_path: Path,
    output_dir: Path,
    vis_threshold: float = 0.5,
    show: bool = False,
) -> tuple[int, list[dict]]:
    """
    Process a single image and save the annotated result.

    Args:
        detector: Face detector instance
        image_path: Path to input image
        output_dir: Directory to save output images
        vis_threshold: Minimum confidence to visualize
        show: Whether to display the result in a window

    Returns:
        Tuple of (number of faces detected, list of face detections)
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image '{image_path}'")
        return 0, []

    # Run detection
    faces = detect_faces(detector, image)

    # Draw detections on image
    if faces:
        bboxes = [face['bbox'] for face in faces]
        scores = [face['confidence'] for face in faces]
        landmarks = [face['landmarks'] for face in faces]

        draw_detections(
            image=image,
            bboxes=bboxes,
            scores=scores,
            landmarks=landmarks,
            vis_threshold=vis_threshold,
            draw_score=True,
            fancy_bbox=True,
        )

    # Save output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_detected{image_path.suffix}"
    cv2.imwrite(str(output_path), image)
    print(f"Saved: {output_path} ({len(faces)} face(s) detected)")

    # Optionally show the result
    if show:
        cv2.imshow('Face Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return len(faces), faces


def print_detection_results(faces: list[dict], image_path: str) -> None:
    """Print detailed detection results to console."""
    print(f"\n{'='*60}")
    print(f"Detection Results for: {image_path}")
    print(f"{'='*60}")
    print(f"Total faces detected: {len(faces)}")

    for i, face in enumerate(faces, 1):
        bbox = face['bbox']
        conf = face['confidence']
        landmarks = face['landmarks']

        print(f"\nFace {i}:")
        print(f"  Confidence: {conf:.3f}")
        print(f"  Bounding Box: [x1={bbox[0]:.1f}, y1={bbox[1]:.1f}, x2={bbox[2]:.1f}, y2={bbox[3]:.1f}]")
        print(f"  Width x Height: {bbox[2] - bbox[0]:.1f} x {bbox[3] - bbox[1]:.1f}")
        print(f"  Landmarks (5 points):")
        landmark_names = ['Left Eye', 'Right Eye', 'Nose', 'Left Mouth', 'Right Mouth']
        for name, (x, y) in zip(landmark_names, landmarks):
            print(f"    {name}: ({x:.1f}, {y:.1f})")


def main():
    parser = argparse.ArgumentParser(
        description='Face Detection Demo - Detect faces in images using UniFace',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python face_detection_demo.py photo.jpg
  python face_detection_demo.py images/ --method scrfd --output results/
  python face_detection_demo.py face.png --threshold 0.7 --show
        """,
    )
    parser.add_argument('input', type=str, help='Input image or directory path')
    parser.add_argument(
        '--method',
        type=str,
        default='retinaface',
        choices=['retinaface', 'scrfd', 'yolov5face'],
        help='Detection method (default: retinaface)',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Detection confidence threshold (default: 0.5)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Output directory for annotated images (default: outputs)',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display results in a window',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed detection information',
    )

    args = parser.parse_args()

    # Validate input path
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist")
        sys.exit(1)

    # Initialize detector
    print(f"Initializing {args.method} detector...")
    detector = create_detector(args.method, args.threshold)
    print("Detector ready!\n")

    output_dir = Path(args.output)

    # Process single image or directory
    if input_path.is_file():
        num_faces, faces = process_image(
            detector,
            input_path,
            output_dir,
            vis_threshold=args.threshold,
            show=args.show,
        )
        if args.verbose and faces:
            print_detection_results(faces, str(input_path))

    elif input_path.is_dir():
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]

        if not image_files:
            print(f"No image files found in '{input_path}'")
            sys.exit(1)

        print(f"Found {len(image_files)} images to process\n")
        total_faces = 0

        for img_path in sorted(image_files):
            num_faces, faces = process_image(
                detector,
                img_path,
                output_dir,
                vis_threshold=args.threshold,
                show=args.show,
            )
            total_faces += num_faces
            if args.verbose and faces:
                print_detection_results(faces, str(img_path))

        print(f"\nTotal: {total_faces} face(s) detected across {len(image_files)} images")
    else:
        print(f"Error: '{input_path}' is not a valid file or directory")
        sys.exit(1)


if __name__ == '__main__':
    main()
