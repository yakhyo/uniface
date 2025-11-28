# 106-point facial landmark detection
# Usage: python run_landmarks.py --image path/to/image.jpg
#        python run_landmarks.py --webcam

import argparse
import os
from pathlib import Path

import cv2

from uniface import SCRFD, Landmark106, RetinaFace


def process_image(detector, landmarker, image_path: str, save_dir: str = 'outputs'):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from '{image_path}'")
        return

    faces = detector.detect(image)
    print(f'Detected {len(faces)} face(s)')

    if not faces:
        return

    for i, face in enumerate(faces):
        bbox = face['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = landmarker.get_landmarks(image, bbox)
        print(f'  Face {i + 1}: {len(landmarks)} landmarks')

        for x, y in landmarks.astype(int):
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

        cv2.putText(
            image,
            f'Face {i + 1}',
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{Path(image_path).stem}_landmarks.jpg')
    cv2.imwrite(output_path, image)
    print(f'Output saved: {output_path}')


def run_webcam(detector, landmarker):
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        print('Cannot open webcam')
        return

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        if not ret:
            break

        faces = detector.detect(frame)

        for face in faces:
            bbox = face['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landmarks = landmarker.get_landmarks(frame, bbox)  # 106 points
            for x, y in landmarks.astype(int):
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        cv2.putText(
            frame,
            f'Faces: {len(faces)}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow('106-Point Landmarks', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Run facial landmark detection')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--webcam', action='store_true', help='Use webcam')
    parser.add_argument('--detector', type=str, default='retinaface', choices=['retinaface', 'scrfd'])
    parser.add_argument('--save_dir', type=str, default='outputs')
    args = parser.parse_args()

    if not args.image and not args.webcam:
        parser.error('Either --image or --webcam must be specified')

    detector = RetinaFace() if args.detector == 'retinaface' else SCRFD()
    landmarker = Landmark106()

    if args.webcam:
        run_webcam(detector, landmarker)
    else:
        process_image(detector, landmarker, args.image, args.save_dir)


if __name__ == '__main__':
    main()
