"""Age and Gender Detection Demo Script"""

import os
import cv2
import argparse
from pathlib import Path

from uniface import RetinaFace, SCRFD, AgeGender
from uniface.visualization import draw_detections


def process_image(detector, age_gender, image_path: str, save_dir: str = "outputs", vis_threshold: float = 0.6):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from '{image_path}'")
        return

    print(f"Processing: {image_path}")

    # Detect faces
    faces = detector.detect(image)
    print(f"  Detected {len(faces)} face(s)")

    if not faces:
        print("  No faces detected")
        return

    # Draw detections
    bboxes = [f['bbox'] for f in faces]
    scores = [f['confidence'] for f in faces]
    landmarks = [f['landmarks'] for f in faces]
    draw_detections(image, bboxes, scores, landmarks, vis_threshold=vis_threshold)

    # Predict and draw age/gender for each face
    for i, face in enumerate(faces):
        gender, age = age_gender.predict(image, face['bbox'])
        print(f"  Face {i+1}: {gender}, {age} years old")

        # Draw age and gender text
        bbox = face['bbox']
        x1, y1 = int(bbox[0]), int(bbox[1])
        text = f"{gender}, {age}y"

        # Background rectangle for text
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - text_height - 10),
                    (x1 + text_width + 10, y1), (0, 255, 0), -1)
        cv2.putText(image, text, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Save result
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"{Path(image_path).stem}_age_gender.jpg")
    cv2.imwrite(output_path, image)
    print(f"Output saved: {output_path}")


def run_webcam(detector, age_gender, vis_threshold: float = 0.6):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Webcam opened")
    print("Press 'q' to quit\n")

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Detect faces
            faces = detector.detect(frame)

            # Draw detections
            bboxes = [f['bbox'] for f in faces]
            scores = [f['confidence'] for f in faces]
            landmarks = [f['landmarks'] for f in faces]
            draw_detections(frame, bboxes, scores, landmarks, vis_threshold=vis_threshold)

            # Predict and draw age/gender for each face
            for face in faces:
                gender, age = age_gender.predict(frame, face['bbox'])

                # Draw age and gender text
                bbox = face['bbox']
                x1, y1 = int(bbox[0]), int(bbox[1])
                text = f"{gender}, {age}y"

                # Background rectangle for text
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - text_height - 10),
                            (x1 + text_width + 10, y1), (0, 255, 0), -1)
                cv2.putText(frame, text, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Add info
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Age & Gender Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames")


def main():
    parser = argparse.ArgumentParser(description="Run age and gender detection")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--webcam", action="store_true", help="Use webcam instead of image")
    parser.add_argument("--detector", type=str, default="retinaface",
                       choices=['retinaface', 'scrfd'], help="Face detector to use")
    parser.add_argument("--threshold", type=float, default=0.6,
                       help="Confidence threshold for visualization")
    parser.add_argument("--save_dir", type=str, default="outputs",
                       help="Directory to save output images")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Validate input
    if not args.image and not args.webcam:
        parser.error("Either --image or --webcam must be specified")

    if args.verbose:
        from uniface import enable_logging
        enable_logging()

    # Initialize models
    print(f"Initializing detector: {args.detector}")
    if args.detector == 'retinaface':
        detector = RetinaFace()
    else:
        detector = SCRFD()

    print("Initializing age/gender model...")
    age_gender = AgeGender()
    print("Models initialized\n")

    # Process
    if args.webcam:
        run_webcam(detector, age_gender, args.threshold)
    else:
        process_image(detector, age_gender, args.image, args.save_dir, args.threshold)


if __name__ == "__main__":
    main()
