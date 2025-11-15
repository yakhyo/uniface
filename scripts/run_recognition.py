import cv2
import argparse
import numpy as np

from uniface.detection import RetinaFace, SCRFD
from uniface.recognition import ArcFace, MobileFace, SphereFace
from uniface.face_utils import compute_similarity


def run_inference(detector, recognizer, image_path: str):
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


def compare_faces(detector, recognizer, image1_path: str, image2_path: str, threshold: float = 0.35):

    # Load images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None or img2 is None:
        print(f"Error: Failed to load images")
        return

    # Detect faces
    faces1 = detector.detect(img1)
    faces2 = detector.detect(img2)

    if not faces1 or not faces2:
        print("Error: No faces detected in one or both images")
        return

    # Get landmarks for first face in each image
    landmarks1 = np.array(faces1[0]['landmarks'])
    landmarks2 = np.array(faces2[0]['landmarks'])

    # Get normalized embeddings
    embedding1 = recognizer.get_normalized_embedding(img1, landmarks1)
    embedding2 = recognizer.get_normalized_embedding(img2, landmarks2)

    # Compute similarity
    similarity = compute_similarity(embedding1, embedding2, normalized=True)
    is_match = similarity > threshold

    print(f"Similarity: {similarity:.4f}")
    print(f"Result: {'Same person' if is_match else 'Different person'}")
    print(f"Threshold: {threshold}")


def main():
    parser = argparse.ArgumentParser(description="Face recognition and comparison.")
    parser.add_argument("--image", type=str, help="Path to single image for embedding extraction.")
    parser.add_argument("--image1", type=str, help="Path to first image for comparison.")
    parser.add_argument("--image2", type=str, help="Path to second image for comparison.")
    parser.add_argument("--threshold", type=float, default=0.35, help="Similarity threshold for face matching.")
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
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        from uniface import enable_logging
        enable_logging()

    print(f"Initializing detector: {args.detector}")
    if args.detector == 'retinaface':
        detector = RetinaFace()
    else:
        detector = SCRFD()

    print(f"Initializing recognizer: {args.recognizer}")
    if args.recognizer == 'arcface':
        recognizer = ArcFace()
    elif args.recognizer == 'mobileface':
        recognizer = MobileFace()
    else:
        recognizer = SphereFace()

    if args.image1 and args.image2:
        # Face comparison mode
        print(f"Comparing faces: {args.image1} vs {args.image2}")
        compare_faces(detector, recognizer, args.image1, args.image2, args.threshold)
    elif args.image:
        # Single image embedding extraction mode
        run_inference(detector, recognizer, args.image)
    else:
        print("Error: Provide either --image for single image processing or --image1 and --image2 for comparison")
        parser.print_help()


if __name__ == "__main__":
    main()
