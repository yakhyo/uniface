# Real-time face search: match webcam faces against a reference image
# Usage: python run_face_search.py --image reference.jpg

import argparse

import cv2
import numpy as np

from uniface.detection import SCRFD, RetinaFace
from uniface.face_utils import compute_similarity
from uniface.recognition import ArcFace, MobileFace, SphereFace


def get_recognizer(name: str):
    if name == 'arcface':
        return ArcFace()
    elif name == 'mobileface':
        return MobileFace()
    else:
        return SphereFace()


def extract_reference_embedding(detector, recognizer, image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f'Failed to load image: {image_path}')

    faces = detector.detect(image)
    if not faces:
        raise RuntimeError('No faces found in reference image.')

    landmarks = faces[0].landmarks
    return recognizer.get_normalized_embedding(image, landmarks)


def run_webcam(detector, recognizer, ref_embedding: np.ndarray, threshold: float = 0.4):
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        raise RuntimeError('Webcam could not be opened.')

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        if not ret:
            break

        faces = detector.detect(frame)

        for face in faces:
            bbox = face.bbox
            landmarks = face.landmarks
            x1, y1, x2, y2 = map(int, bbox)

            embedding = recognizer.get_normalized_embedding(frame, landmarks)
            sim = compute_similarity(ref_embedding, embedding)  # compare with reference

            # green = match, red = unknown
            label = f'Match ({sim:.2f})' if sim > threshold else f'Unknown ({sim:.2f})'
            color = (0, 255, 0) if sim > threshold else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Face search using a reference image')
    parser.add_argument('--image', type=str, required=True, help='Reference face image')
    parser.add_argument('--threshold', type=float, default=0.4, help='Match threshold')
    parser.add_argument('--detector', type=str, default='scrfd', choices=['retinaface', 'scrfd'])
    parser.add_argument(
        '--recognizer',
        type=str,
        default='arcface',
        choices=['arcface', 'mobileface', 'sphereface'],
    )
    args = parser.parse_args()

    detector = RetinaFace() if args.detector == 'retinaface' else SCRFD()
    recognizer = get_recognizer(args.recognizer)

    print(f'Loading reference: {args.image}')
    ref_embedding = extract_reference_embedding(detector, recognizer, args.image)

    run_webcam(detector, recognizer, ref_embedding, args.threshold)


if __name__ == '__main__':
    main()
