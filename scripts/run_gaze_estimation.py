# Gaze estimation on detected faces
# Usage: python run_gaze_estimation.py --image path/to/image.jpg
#        python run_gaze_estimation.py --webcam

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from uniface import RetinaFace
from uniface.gaze import MobileGaze
from uniface.visualization import draw_gaze


def process_image(detector, gaze_estimator, image_path: str, save_dir: str = 'outputs'):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from '{image_path}'")
        return

    faces = detector.detect(image)
    print(f'Detected {len(faces)} face(s)')

    for i, face in enumerate(faces):
        bbox = face['bbox']
        x1, y1, x2, y2 = map(int, bbox[:4])
        face_crop = image[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        pitch, yaw = gaze_estimator.estimate(face_crop)
        print(f'  Face {i + 1}: pitch={np.degrees(pitch):.1f}°, yaw={np.degrees(yaw):.1f}°')

        # Draw both bbox and gaze arrow with angle text
        draw_gaze(image, bbox, pitch, yaw, draw_angles=True)

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{Path(image_path).stem}_gaze.jpg')
    cv2.imwrite(output_path, image)
    print(f'Output saved: {output_path}')


def run_webcam(detector, gaze_estimator):
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

            pitch, yaw = gaze_estimator.estimate(face_crop)
            # Draw both bbox and gaze arrow
            draw_gaze(frame, bbox, pitch, yaw)

        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Gaze Estimation', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Run gaze estimation')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--webcam', action='store_true', help='Use webcam')
    parser.add_argument('--save_dir', type=str, default='outputs')
    args = parser.parse_args()

    if not args.image and not args.webcam:
        parser.error('Either --image or --webcam must be specified')

    detector = RetinaFace()
    gaze_estimator = MobileGaze()

    if args.webcam:
        run_webcam(detector, gaze_estimator)
    else:
        process_image(detector, gaze_estimator, args.image, args.save_dir)


if __name__ == '__main__':
    main()
