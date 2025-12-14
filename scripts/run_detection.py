# Face detection on image or webcam
# Usage: python run_detection.py --image path/to/image.jpg
#        python run_detection.py --webcam

import argparse
import os

import cv2

from uniface.detection import SCRFD, RetinaFace, YOLOv5Face
from uniface.visualization import draw_detections


def process_image(detector, image_path: str, threshold: float = 0.6, save_dir: str = 'outputs'):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from '{image_path}'")
        return

    faces = detector.detect(image)

    if faces:
        bboxes = [face['bbox'] for face in faces]
        scores = [face['confidence'] for face in faces]
        landmarks = [face['landmarks'] for face in faces]
        draw_detections(image, bboxes, scores, landmarks, vis_threshold=threshold)

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{os.path.splitext(os.path.basename(image_path))[0]}_out.jpg')
    cv2.imwrite(output_path, image)
    print(f'Output saved: {output_path}')


def run_webcam(detector, threshold: float = 0.6):
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

        # unpack face data for visualization
        bboxes = [f['bbox'] for f in faces]
        scores = [f['confidence'] for f in faces]
        landmarks = [f['landmarks'] for f in faces]
        draw_detections(
            image=frame,
            bboxes=bboxes,
            scores=scores,
            landmarks=landmarks,
            vis_threshold=threshold,
            draw_score=True,
            fancy_bbox=True,
        )

        cv2.putText(
            frame,
            f'Faces: {len(faces)}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Run face detection')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--webcam', action='store_true', help='Use webcam')
    parser.add_argument('--method', type=str, default='retinaface', choices=['retinaface', 'scrfd', 'yolov5face'])
    parser.add_argument('--threshold', type=float, default=0.25, help='Visualization threshold')
    parser.add_argument('--save_dir', type=str, default='outputs')
    args = parser.parse_args()

    if not args.image and not args.webcam:
        parser.error('Either --image or --webcam must be specified')

    if args.method == 'retinaface':
        detector = RetinaFace()
    elif args.method == 'scrfd':
        detector = SCRFD()
    else:
        from uniface.constants import YOLOv5FaceWeights

        detector = YOLOv5Face(model_name=YOLOv5FaceWeights.YOLOV5M)

    if args.webcam:
        run_webcam(detector, args.threshold)
    else:
        process_image(detector, args.image, args.threshold, args.save_dir)


if __name__ == '__main__':
    main()
