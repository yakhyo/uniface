# Face anonymization/blurring for privacy
# Usage: python run_anonymization.py --image path/to/image.jpg --method pixelate
#        python run_anonymization.py --webcam --method gaussian

import argparse
import os

import cv2

from uniface import RetinaFace
from uniface.privacy import BlurFace


def process_image(
    detector,
    blurrer: BlurFace,
    image_path: str,
    save_dir: str = 'outputs',
    show_detections: bool = False,
):
    """Process a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from '{image_path}'")
        return

    # Detect faces
    faces = detector.detect(image)
    print(f'Detected {len(faces)} face(s)')

    # Optionally draw detection boxes before blurring
    if show_detections and faces:
        from uniface.visualization import draw_detections

        preview = image.copy()
        bboxes = [face['bbox'] for face in faces]
        scores = [face['confidence'] for face in faces]
        landmarks = [face['landmarks'] for face in faces]
        draw_detections(preview, bboxes, scores, landmarks)

        # Show preview
        cv2.imshow('Detections (Press any key to continue)', preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Anonymize faces
    if faces:
        anonymized = blurrer.anonymize(image, faces)
    else:
        anonymized = image

    # Save output
    os.makedirs(save_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(save_dir, f'{basename}_anonymized.jpg')
    cv2.imwrite(output_path, anonymized)
    print(f'Output saved: {output_path}')


def run_webcam(detector, blurrer: BlurFace):
    """Run real-time anonymization on webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open webcam')
        return

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        if not ret:
            break

        # Detect and anonymize
        faces = detector.detect(frame)
        if faces:
            frame = blurrer.anonymize(frame, faces, inplace=True)

        # Display info
        cv2.putText(
            frame,
            f'Faces blurred: {len(faces)} | Method: {blurrer.method}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow('Face Anonymization (Press q to quit)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Face anonymization using various blur methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Anonymize image with pixelation (default)
  python run_anonymization.py --image photo.jpg

  # Use Gaussian blur with custom strength
  python run_anonymization.py --image photo.jpg --method gaussian --blur-strength 5.0

  # Real-time webcam anonymization
  python run_anonymization.py --webcam --method pixelate

  # Black boxes for maximum privacy
  python run_anonymization.py --image photo.jpg --method blackout

  # Custom pixelation intensity
  python run_anonymization.py --image photo.jpg --method pixelate --pixel-blocks 5
        """,
    )

    # Input/output
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for real-time anonymization')
    parser.add_argument('--save-dir', type=str, default='outputs', help='Output directory (default: outputs)')

    # Blur method
    parser.add_argument(
        '--method',
        type=str,
        default='pixelate',
        choices=['gaussian', 'pixelate', 'blackout', 'elliptical', 'median'],
        help='Blur method (default: pixelate)',
    )

    # Method-specific parameters
    parser.add_argument(
        '--blur-strength',
        type=float,
        default=3.0,
        help='Blur strength for gaussian/elliptical/median (default: 3.0)',
    )
    parser.add_argument(
        '--pixel-blocks',
        type=int,
        default=20,
        help='Number of pixel blocks for pixelate (default: 10, lower=more pixelated)',
    )
    parser.add_argument(
        '--color',
        type=str,
        default='0,0,0',
        help='Fill color for blackout as R,G,B (default: 0,0,0 for black)',
    )
    parser.add_argument('--margin', type=int, default=20, help='Margin for elliptical blur (default: 20)')

    # Detection
    parser.add_argument(
        '--conf-thresh',
        type=float,
        default=0.5,
        help='Detection confidence threshold (default: 0.5)',
    )

    # Visualization
    parser.add_argument(
        '--show-detections',
        action='store_true',
        help='Show detection boxes before blurring (image mode only)',
    )

    args = parser.parse_args()

    # Validate input
    if not args.image and not args.webcam:
        parser.error('Either --image or --webcam must be specified')

    # Parse color
    color_values = [int(x) for x in args.color.split(',')]
    if len(color_values) != 3:
        parser.error('--color must be in format R,G,B (e.g., 0,0,0)')
    color = tuple(color_values)

    # Initialize detector
    print(f'Initializing face detector (conf_thresh={args.conf_thresh})...')
    detector = RetinaFace(conf_thresh=args.conf_thresh)

    # Initialize blurrer
    print(f'Initializing blur method: {args.method}')
    blurrer = BlurFace(
        method=args.method,
        blur_strength=args.blur_strength,
        pixel_blocks=args.pixel_blocks,
        color=color,
        margin=args.margin,
    )

    # Run
    if args.webcam:
        run_webcam(detector, blurrer)
    else:
        process_image(detector, blurrer, args.image, args.save_dir, args.show_detections)


if __name__ == '__main__':
    main()

