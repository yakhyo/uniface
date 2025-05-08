import cv2
import argparse
import numpy as np

from uniface.detection import RetinaFace
from uniface.constants import RetinaFaceWeights
from uniface.recognition import ArcFace
from uniface.face_utils import compute_similarity


def extract_reference_embedding(detector, recognizer, image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    boxes, landmarks = detector.detect(image)
    if len(boxes) == 0:
        raise RuntimeError("No faces found in reference image.")

    embedding = recognizer.get_embedding(image, landmarks[0])
    return embedding


def run_video(detector, recognizer, ref_embedding, threshold=0.30):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam could not be opened.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, landmarks = detector.detect(frame)

        for box, lm in zip(boxes, landmarks):
            x1, y1, x2, y2 = map(int, box[:4])
            embedding = recognizer.get_embedding(frame, lm)
            sim = compute_similarity(ref_embedding, embedding)
            label = f"Match ({sim:.2f})" if sim > threshold else f"Unknown ({sim:.2f})"
            color = (0, 255, 0) if sim > threshold else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Face recognition using a reference image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the reference face image.")
    parser.add_argument("--model", type=str, default="MNET_V2",
                        choices=[m.name for m in RetinaFaceWeights], help="Face detector model.")
    args = parser.parse_args()

    detector = RetinaFace(model_name=RetinaFaceWeights[args.model])
    recognizer = ArcFace()
    ref_embedding = extract_reference_embedding(detector, recognizer, args.image)
    run_video(detector, recognizer, ref_embedding)


if __name__ == "__main__":
    main()
