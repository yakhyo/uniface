import cv2
import argparse
import numpy as np

from uniface.detection import RetinaFace
from uniface.constants import RetinaFaceWeights
from uniface.recognition import ArcFace


def run_inference(detector, recognizer, image_path):
    """
    Detect faces and extract embeddings from a single image.

    Args:
        detector (RetinaFace): Initialized face detector.
        recognizer (ArcFace): Face recognition model.
        image_path (str): Path to the input image.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from '{image_path}'")
        return

    boxes, landmarks = detector.detect(image)

    if len(boxes) == 0:
        print("No faces detected.")
        return

    print(f"Detected {len(boxes)} face(s). Extracting embeddings...")

    for i, landmark in enumerate(landmarks):
        embedding = recognizer.get_embedding(image, landmark)
        norm = np.linalg.norm(embedding)
        print(f"\nFace {i} embedding (L2 norm = {norm:.4f}):")
        print(embedding)


def main():
    parser = argparse.ArgumentParser(description="Extract face embeddings from a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument(
        "--model",
        type=str,
        default="MNET_V2",
        choices=[m.name for m in RetinaFaceWeights],
        help="RetinaFace model variant to use."
    )

    args = parser.parse_args()

    detector = RetinaFace(model_name=RetinaFaceWeights[args.model])
    recognizer = ArcFace()

    run_inference(detector, recognizer, args.image)


if __name__ == "__main__":
    main()
