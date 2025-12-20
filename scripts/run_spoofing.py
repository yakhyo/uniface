# Face Anti-Spoofing Detection
# Usage:
#   Image: python run_spoofing.py --image path/to/image.jpg
#   Video: python run_spoofing.py --video path/to/video.mp4
#   Webcam: python run_spoofing.py --source 0

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from uniface import RetinaFace
from uniface.constants import MiniFASNetWeights
from uniface.spoofing import create_spoofer


def draw_spoofing_result(
    image: np.ndarray,
    bbox: list,
    label_idx: int,
    score: float,
    thickness: int = 2,
) -> None:
    """Draw bounding box with anti-spoofing result.

    Args:
        image: Input image to draw on.
        bbox: Bounding box in [x1, y1, x2, y2] format.
        label_idx: Prediction label index (0 = Fake, 1 = Real).
        score: Confidence score (0.0 to 1.0).
        thickness: Line thickness for bounding box.
    """
    x1, y1, x2, y2 = map(int, bbox[:4])

    # Color based on result (green for real, red for fake)
    is_real = label_idx == 1
    color = (0, 255, 0) if is_real else (0, 0, 255)

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Prepare label
    label = 'Real' if is_real else 'Fake'
    text = f'{label}: {score:.1%}'

    # Draw label background
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)

    # Draw label text
    cv2.putText(image, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def process_image(detector, spoofer, image_path: str, save_dir: str = 'outputs') -> None:
    """Process a single image for face anti-spoofing detection."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from '{image_path}'")
        return

    # Detect faces
    faces = detector.detect(image)
    print(f'Detected {len(faces)} face(s)')

    if not faces:
        print('No faces detected in the image.')
        return

    # Run anti-spoofing on each face
    for i, face in enumerate(faces, 1):
        label_idx, score = spoofer.predict(image, face['bbox'])
        # label_idx: 0 = Fake, 1 = Real
        label = 'Real' if label_idx == 1 else 'Fake'
        print(f'  Face {i}: {label} ({score:.1%})')

        # Draw result on image
        draw_spoofing_result(image, face['bbox'], label_idx, score)

    # Save output
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{Path(image_path).stem}_spoofing.jpg')
    cv2.imwrite(output_path, image)
    print(f'Output saved: {output_path}')


def process_video(detector, spoofer, source, save_dir: str = 'outputs') -> None:
    """Process video or webcam stream for face anti-spoofing detection."""
    # Handle webcam or video file
    if isinstance(source, int) or source.isdigit():
        cap = cv2.VideoCapture(int(source))
        is_webcam = True
        output_name = 'webcam_spoofing.mp4'
    else:
        cap = cv2.VideoCapture(source)
        is_webcam = False
        output_name = f'{Path(source).stem}_spoofing.mp4'

    if not cap.isOpened():
        print(f'Error: Failed to open video source: {source}')
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if not is_webcam else 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup video writer
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, output_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Processing video... Press 'q' to quit")
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Detect faces
            faces = detector.detect(frame)

            # Run anti-spoofing on each face
            for face in faces:
                label_idx, score = spoofer.predict(frame, face['bbox'])
                draw_spoofing_result(frame, face['bbox'], label_idx, score)

            # Write frame
            writer.write(frame)

            # Display frame
            cv2.imshow('Face Anti-Spoofing', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Stopped by user.')
                break

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f'Processed {frame_count} frames')
    if not is_webcam:
        print(f'Output saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Face Anti-Spoofing Detection')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--source', type=str, help='Video source (0 for webcam)')
    parser.add_argument(
        '--model',
        type=str,
        default='v2',
        choices=['v1se', 'v2'],
        help='Model variant: v1se or v2 (default: v2)',
    )
    parser.add_argument('--scale', type=float, default=None, help='Custom crop scale (default: auto)')
    parser.add_argument('--save_dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()

    # Check that at least one input source is provided
    if not any([args.image, args.video, args.source]):
        parser.print_help()
        print('\nError: Please provide --image, --video, or --source')
        return

    # Select model variant
    model_name = MiniFASNetWeights.V1SE if args.model == 'v1se' else MiniFASNetWeights.V2

    # Initialize models
    print(f'Initializing models (MiniFASNet {args.model.upper()})...')
    detector = RetinaFace()
    spoofer = create_spoofer(model_name=model_name, scale=args.scale)

    # Process input
    if args.image:
        if not os.path.exists(args.image):
            print(f'Error: Image not found: {args.image}')
            return
        process_image(detector, spoofer, args.image, args.save_dir)

    elif args.video:
        if not os.path.exists(args.video):
            print(f'Error: Video not found: {args.video}')
            return
        process_video(detector, spoofer, args.video, args.save_dir)

    elif args.source:
        process_video(detector, spoofer, args.source, args.save_dir)


if __name__ == '__main__':
    main()
