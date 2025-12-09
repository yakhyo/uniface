# Face analysis using FaceAnalyzer
# Usage: python run_face_analyzer.py --image path/to/image.jpg

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from uniface import AgeGender, ArcFace, FaceAnalyzer, RetinaFace
from uniface.visualization import draw_detections


def draw_face_info(image, face, face_id):
    """Draw face ID and attributes above bounding box."""
    x1, y1, x2, y2 = map(int, face.bbox)
    lines = [f'ID: {face_id}', f'Conf: {face.confidence:.2f}']
    if face.age and face.sex:
        lines.append(f'{face.sex}, {face.age}y')

    for i, line in enumerate(lines):
        y_pos = y1 - 10 - (len(lines) - 1 - i) * 25
        if y_pos < 20:
            y_pos = y2 + 20 + i * 25
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y_pos - th - 5), (x1 + tw + 10, y_pos + 5), (0, 255, 0), -1)
        cv2.putText(image, line, (x1 + 5, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def process_image(analyzer, image_path: str, save_dir: str = 'outputs', show_similarity: bool = True):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from '{image_path}'")
        return

    faces = analyzer.analyze(image)
    print(f'Detected {len(faces)} face(s)')

    if not faces:
        return

    for i, face in enumerate(faces, 1):
        info = f'  Face {i}: {face.sex}, {face.age}y' if face.age and face.sex else f'  Face {i}'
        if face.embedding is not None:
            info += f' (embedding: {face.embedding.shape})'
        print(info)

    if show_similarity and len(faces) >= 2:
        print('\nSimilarity Matrix:')
        n = len(faces)
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    sim_matrix[i][j] = 1.0
                else:
                    sim = faces[i].compute_similarity(faces[j])
                    sim_matrix[i][j] = sim
                    sim_matrix[j][i] = sim

        print('     ', end='')
        for i in range(n):
            print(f'  F{i + 1:2d}  ', end='')
        print('\n     ' + '-' * (7 * n))

        for i in range(n):
            print(f'F{i + 1:2d} | ', end='')
            for j in range(n):
                print(f'{sim_matrix[i][j]:6.3f} ', end='')
            print()

        pairs = [(i, j, sim_matrix[i][j]) for i in range(n) for j in range(i + 1, n)]
        pairs.sort(key=lambda x: x[2], reverse=True)

        print('\nTop matches (>0.4 = same person):')
        for i, j, sim in pairs[:3]:
            status = 'Same' if sim > 0.4 else 'Different'
            print(f'  Face {i + 1} ↔ Face {j + 1}: {sim:.3f} ({status})')

    bboxes = [f.bbox for f in faces]
    scores = [f.confidence for f in faces]
    landmarks = [f.landmarks for f in faces]
    draw_detections(image=image, bboxes=bboxes, scores=scores, landmarks=landmarks, fancy_bbox=True)

    for i, face in enumerate(faces, 1):
        draw_face_info(image, face, i)

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'{Path(image_path).stem}_analysis.jpg')
    cv2.imwrite(output_path, image)
    print(f'Output saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Face analysis with detection, recognition, and attributes')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--save_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--no-similarity', action='store_true', help='Skip similarity matrix computation')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f'Error: Image not found: {args.image}')
        return

    detector = RetinaFace()
    recognizer = ArcFace()
    age_gender = AgeGender()
    analyzer = FaceAnalyzer(detector, recognizer, age_gender)

    process_image(analyzer, args.image, args.save_dir, show_similarity=not args.no_similarity)


if __name__ == '__main__':
    main()
