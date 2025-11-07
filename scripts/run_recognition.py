import cv2
import argparse
import numpy as np

# Use the new high-level factory functions for consistency
from uniface.detection import create_detector
from uniface.recognition import create_recognizer


def run_inference(detector, recognizer, image_path: str):
    """
    Detect faces and extract embeddings from a single image.

    Args:
        detector: Initialized face detector.
        recognizer: Initialized face recognition model.
        image_path (str): Path to the input image.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from '{image_path}'")
        return

    faces = detector.detect(image)

    if not faces:
        print("No faces detected.")
        return

    print(f"Detected {len(faces)} face(s). Extracting embeddings for the first face...")

    # Process the first detected face
    first_face = faces[0]
    landmarks = np.array(first_face['landmarks'])  # Convert landmarks to numpy array

    # Extract embedding using the landmarks from the face dictionary
    embedding = recognizer.get_embedding(image, landmarks)
    norm_embedding = recognizer.get_normalized_embedding(image, landmarks)

    # Print some info about the embeddings
    print(f"  - Embedding shape: {embedding.shape}")
    print(f"  - L2 norm of unnormalized embedding: {np.linalg.norm(embedding):.4f}")
    print(f"  - L2 norm of normalized embedding: {np.linalg.norm(norm_embedding):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Extract face embeddings from a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument(
        "--detector",
        type=str,
        default="retinaface",
        choices=['retinaface', 'scrfd'],
        help="Face detection method to use."
    )
    parser.add_argument(
        "--recognizer",
        type=str,
        default="arcface",
        choices=['arcface', 'mobileface', 'sphereface'],
        help="Face recognition method to use."
    )

    args = parser.parse_args()

    print(f"Initializing detector: {args.detector}")
    detector = create_detector(method=args.detector)

    print(f"Initializing recognizer: {args.recognizer}")
    recognizer = create_recognizer(method=args.recognizer)

    run_inference(detector, recognizer, args.image)


if __name__ == "__main__":
    main()
