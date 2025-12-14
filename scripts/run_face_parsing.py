# Face parsing on detected faces
# Usage: python run_face_parsing.py --image path/to/image.jpg
#        python run_face_parsing.py --webcam

import argparse
import os
from pathlib import Path

import cv2

from uniface import RetinaFace
from uniface.parsing import BiSeNet
from uniface.constants import ParsingWeights
from uniface.parsing.utils import vis_parsing_maps


def process_image(detector, parser, image_path: str, save_dir: str = 'outputs'):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from '{image_path}'")
        return

    faces = detector.detect(image)
    print(f'Detected {len(faces)} face(s)')

    result_image = image.copy()

    for i, face in enumerate(faces):
        bbox = face['bbox']
        x1, y1, x2, y2 = map(int, bbox[:4])
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

        # Draw bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{Path(image_path).stem}_parsing.jpg')
    cv2.imwrite(output_path, result_image)
    print(f'Output saved: {output_path}')


def run_webcam(detector, parser):
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
            bbox = face['bbox']
            x1, y1, x2, y2 = map(int, bbox[:4])
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

            # Draw bounding box
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
    args = parser_arg.parse_args()

    if not args.image and not args.webcam:
        parser_arg.error('Either --image or --webcam must be specified')

    detector = RetinaFace()
    parser = BiSeNet(model_name=ParsingWeights.RESNET34)

    if args.webcam:
        run_webcam(detector, parser)
    else:
        process_image(detector, parser, args.image, args.save_dir)


if __name__ == '__main__':
    main()
