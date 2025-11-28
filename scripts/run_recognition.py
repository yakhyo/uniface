# Face recognition: extract embeddings or compare two faces
# Usage: python run_recognition.py --image path/to/image.jpg
#        python run_recognition.py --image1 face1.jpg --image2 face2.jpg

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


def run_inference(detector, recognizer, image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from '{image_path}'")
        return

    faces = detector.detect(image)
    if not faces:
        print('No faces detected.')
        return

    print(f'Detected {len(faces)} face(s). Extracting embedding for the first face...')

    landmarks = faces[0]['landmarks']  # 5-point landmarks for alignment (already np.ndarray)
    embedding = recognizer.get_embedding(image, landmarks)
    norm_embedding = recognizer.get_normalized_embedding(image, landmarks)  # L2 normalized

    print(f'  Embedding shape: {embedding.shape}')
    print(f'  L2 norm (raw): {np.linalg.norm(embedding):.4f}')
    print(f'  L2 norm (normalized): {np.linalg.norm(norm_embedding):.4f}')


def compare_faces(detector, recognizer, image1_path: str, image2_path: str, threshold: float = 0.35):
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None or img2 is None:
        print('Error: Failed to load one or both images')
        return

    faces1 = detector.detect(img1)
    faces2 = detector.detect(img2)

    if not faces1 or not faces2:
        print('Error: No faces detected in one or both images')
        return

    landmarks1 = faces1[0]['landmarks']
    landmarks2 = faces2[0]['landmarks']

    embedding1 = recognizer.get_normalized_embedding(img1, landmarks1)
    embedding2 = recognizer.get_normalized_embedding(img2, landmarks2)

    # cosine similarity for normalized embeddings
    similarity = compute_similarity(embedding1, embedding2, normalized=True)
    is_match = similarity > threshold

    print(f'Similarity: {similarity:.4f}')
    print(f'Result: {"Same person" if is_match else "Different person"} (threshold: {threshold})')


def main():
    parser = argparse.ArgumentParser(description='Face recognition and comparison')
    parser.add_argument('--image', type=str, help='Single image for embedding extraction')
    parser.add_argument('--image1', type=str, help='First image for comparison')
    parser.add_argument('--image2', type=str, help='Second image for comparison')
    parser.add_argument('--threshold', type=float, default=0.35, help='Similarity threshold')
    parser.add_argument('--detector', type=str, default='retinaface', choices=['retinaface', 'scrfd'])
    parser.add_argument(
        '--recognizer',
        type=str,
        default='arcface',
        choices=['arcface', 'mobileface', 'sphereface'],
    )
    args = parser.parse_args()

    detector = RetinaFace() if args.detector == 'retinaface' else SCRFD()
    recognizer = get_recognizer(args.recognizer)

    if args.image1 and args.image2:
        compare_faces(detector, recognizer, args.image1, args.image2, args.threshold)
    elif args.image:
        run_inference(detector, recognizer, args.image)
    else:
        print('Error: Provide --image or both --image1 and --image2')
        parser.print_help()


if __name__ == '__main__':
    main()
