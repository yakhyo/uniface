import argparse

import cv2
import numpy as np

from uniface.detection import RetinaFace, SCRFD
from uniface.face_utils import compute_similarity
from uniface.recognition import ArcFace, MobileFace, SphereFace


def extract_reference_embedding(detector, recognizer, image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    faces = detector.detect(image)
    if not faces:
        raise RuntimeError("No faces found in reference image.")

    # Get landmarks from the first detected face dictionary
    landmarks = np.array(faces[0]["landmarks"])

    # Use normalized embedding for more reliable similarity comparison
    embedding = recognizer.get_normalized_embedding(image, landmarks)
    return embedding


def run_video(detector, recognizer, ref_embedding: np.ndarray, threshold: float = 0.4):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam could not be opened.")
    print("Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect(frame)

        # Loop through each detected face
        for face in faces:
            # Extract bbox and landmarks from the dictionary
            bbox = face["bbox"]
            landmarks = np.array(face["landmarks"])

            x1, y1, x2, y2 = map(int, bbox)

            # Get the normalized embedding for the current face
            embedding = recognizer.get_normalized_embedding(frame, landmarks)

            # Compare with the reference embedding
            sim = compute_similarity(ref_embedding, embedding)

            # Draw results
            label = f"Match ({sim:.2f})" if sim > threshold else f"Unknown ({sim:.2f})"
            color = (0, 255, 0) if sim > threshold else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Face recognition using a reference image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the reference face image.")
    parser.add_argument(
        "--detector", type=str, default="scrfd", choices=["retinaface", "scrfd"], help="Face detection method."
    )
    parser.add_argument(
        "--recognizer",
        type=str,
        default="arcface",
        choices=["arcface", "mobileface", "sphereface"],
        help="Face recognition method.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if args.verbose:
        from uniface import enable_logging

        enable_logging()

    print("Initializing models...")
    if args.detector == 'retinaface':
        detector = RetinaFace()
    else:
        detector = SCRFD()

    if args.recognizer == 'arcface':
        recognizer = ArcFace()
    elif args.recognizer == 'mobileface':
        recognizer = MobileFace()
    else:
        recognizer = SphereFace()

    print("Extracting reference embedding...")
    ref_embedding = extract_reference_embedding(detector, recognizer, args.image)

    run_video(detector, recognizer, ref_embedding)


if __name__ == "__main__":
    main()
