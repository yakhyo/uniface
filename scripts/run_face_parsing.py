# Face parsing on detected faces
# Usage: python run_face_parsing.py --image path/to/image.jpg
#        python run_face_parsing.py --webcam

import argparse
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from uniface import RetinaFace
from uniface.constants import ParsingWeights
from uniface.parsing import BiSeNet
from uniface.visualization import vis_parsing_maps


def expand_bbox(
    bbox: np.ndarray,
    image_shape: Tuple[int, int],
    expand_ratio: float = 0.2,
    expand_top_ratio: float = 0.4,
) -> Tuple[int, int, int, int]:
    """
    Expand bounding box to include full head region for face parsing.

    Face detection typically returns tight face boxes, but face parsing
    requires the full head including hair, ears, and neck.

    Args:
        bbox: Original bounding box [x1, y1, x2, y2].
        image_shape: Image dimensions as (height, width).
        expand_ratio: Expansion ratio for left, right, and bottom (default: 0.2 = 20%).
        expand_top_ratio: Expansion ratio for top to capture hair/forehead (default: 0.4 = 40%).

    Returns:
        Tuple[int, int, int, int]: Expanded bbox (x1, y1, x2, y2) clamped to image bounds.
    """
    x1, y1, x2, y2 = map(int, bbox[:4])
    height, width = image_shape[:2]

    # Calculate face dimensions
    face_width = x2 - x1
    face_height = y2 - y1

    # Calculate expansion amounts
    expand_x = int(face_width * expand_ratio)
    expand_y_bottom = int(face_height * expand_ratio)
    expand_y_top = int(face_height * expand_top_ratio)

    # Expand and clamp to image boundaries
    new_x1 = max(0, x1 - expand_x)
    new_y1 = max(0, y1 - expand_y_top)
    new_x2 = min(width, x2 + expand_x)
    new_y2 = min(height, y2 + expand_y_bottom)

    return new_x1, new_y1, new_x2, new_y2


def process_image(detector, parser, image_path: str, save_dir: str = 'outputs', expand_ratio: float = 0.2):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from '{image_path}'")
        return

    faces = detector.detect(image)
    print(f'Detected {len(faces)} face(s)')

    result_image = image.copy()

    for i, face in enumerate(faces):
        # Expand bbox to include full head for parsing
        x1, y1, x2, y2 = expand_bbox(face.bbox, image.shape, expand_ratio=expand_ratio)
        face_crop = image[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        # Parse the face
        mask = parser.parse(face_crop)
        print(f'  Face {i + 1}: parsed with {len(set(mask.flatten()))} unique classes')

        # Visualize the parsing result
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        vis_result = vis_parsing_maps(face_crop_rgb, mask, save_image=False)

        # Place the visualization back on the original image
        result_image[y1:y2, x1:x2] = vis_result

        # Draw expanded bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{Path(image_path).stem}_parsing.jpg')
    cv2.imwrite(output_path, result_image)
    print(f'Output saved: {output_path}')


def run_webcam(detector, parser, expand_ratio: float = 0.2):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open webcam')
        return

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        faces = detector.detect(frame)

        for face in faces:
            # Expand bbox to include full head for parsing
            x1, y1, x2, y2 = expand_bbox(face.bbox, frame.shape, expand_ratio=expand_ratio)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            # Parse the face
            mask = parser.parse(face_crop)

            # Visualize the parsing result
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            vis_result = vis_parsing_maps(face_crop_rgb, mask, save_image=False)

            # Place the visualization back on the frame
            frame[y1:y2, x1:x2] = vis_result

            # Draw expanded bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Parsing', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser_arg = argparse.ArgumentParser(description='Run face parsing')
    parser_arg.add_argument('--image', type=str, help='Path to input image')
    parser_arg.add_argument('--webcam', action='store_true', help='Use webcam')
    parser_arg.add_argument('--save_dir', type=str, default='outputs')
    parser_arg.add_argument(
        '--model', type=str, default=ParsingWeights.RESNET18, choices=[ParsingWeights.RESNET18, ParsingWeights.RESNET34]
    )
    parser_arg.add_argument(
        '--expand-ratio',
        type=float,
        default=0.2,
        help='Bbox expansion ratio for full head coverage (default: 0.2 = 20%%)',
    )
    args = parser_arg.parse_args()

    if not args.image and not args.webcam:
        parser_arg.error('Either --image or --webcam must be specified')

    detector = RetinaFace()
    parser = BiSeNet(model_name=ParsingWeights.RESNET34)

    if args.webcam:
        run_webcam(detector, parser, expand_ratio=args.expand_ratio)
    else:
        process_image(detector, parser, args.image, args.save_dir, expand_ratio=args.expand_ratio)


if __name__ == '__main__':
    main()
