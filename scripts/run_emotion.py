# Emotion detection on detected faces
# Usage: python run_emotion.py --image path/to/image.jpg
#        python run_emotion.py --webcam

import argparse
import os
from pathlib import Path

import cv2

from uniface import SCRFD, Emotion, RetinaFace
from uniface.visualization import draw_detections


def draw_emotion_label(image, bbox, emotion: str, confidence: float):
    """Draw emotion label above the bounding box."""
    x1, y1 = int(bbox[0]), int(bbox[1])
    text = f'{emotion} ({confidence:.2f})'
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw + 10, y1), (255, 0, 0), -1)
    cv2.putText(image, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def process_image(
    detector,
    emotion_predictor,
    image_path: str,
    save_dir: str = 'outputs',
    threshold: float = 0.6,
):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from '{image_path}'")
        return

    faces = detector.detect(image)
    print(f'Detected {len(faces)} face(s)')

    if not faces:
        return

    bboxes = [f['bbox'] for f in faces]
    scores = [f['confidence'] for f in faces]
    landmarks = [f['landmarks'] for f in faces]
    draw_detections(
        image=image, bboxes=bboxes, scores=scores, landmarks=landmarks, vis_threshold=threshold, fancy_bbox=True
    )

    for i, face in enumerate(faces):
        emotion, confidence = emotion_predictor.predict(image, face['landmarks'])
        print(f'  Face {i + 1}: {emotion} (confidence: {confidence:.3f})')
        draw_emotion_label(image, face['bbox'], emotion, confidence)

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{Path(image_path).stem}_emotion.jpg')
    cv2.imwrite(output_path, image)
    print(f'Output saved: {output_path}')


def run_webcam(detector, emotion_predictor, threshold: float = 0.6):
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
        draw_detections(frame, bboxes, scores, landmarks, vis_threshold=threshold)

        for face in faces:
            emotion, confidence = emotion_predictor.predict(frame, face['landmarks'])
            draw_emotion_label(frame, face['bbox'], emotion, confidence)

        cv2.putText(
            frame,
            f'Faces: {len(faces)}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Run emotion detection')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--webcam', action='store_true', help='Use webcam')
    parser.add_argument('--detector', type=str, default='retinaface', choices=['retinaface', 'scrfd'])
    parser.add_argument('--threshold', type=float, default=0.6, help='Visualization threshold')
    parser.add_argument('--save_dir', type=str, default='outputs')
    args = parser.parse_args()

    if not args.image and not args.webcam:
        parser.error('Either --image or --webcam must be specified')

    detector = RetinaFace() if args.detector == 'retinaface' else SCRFD()
    emotion_predictor = Emotion()

    if args.webcam:
        run_webcam(detector, emotion_predictor, args.threshold)
    else:
        process_image(detector, emotion_predictor, args.image, args.save_dir, args.threshold)


if __name__ == '__main__':
    main()
