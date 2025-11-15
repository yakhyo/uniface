"""Facial Landmark Detection Demo Script"""

import os
import cv2
import argparse
from pathlib import Path

from uniface import RetinaFace, SCRFD, Landmark106


def process_image(detector, landmarker, image_path: str, save_dir: str = "outputs"):
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

    # Process each face
    for i, face in enumerate(faces):
        # Draw bounding box
        bbox = face['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Get and draw 106 landmarks
        landmarks = landmarker.get_landmarks(image, bbox)
        print(f"  Face {i+1}: Extracted {len(landmarks)} landmarks")

        for x, y in landmarks.astype(int):
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

        # Add face count
        cv2.putText(image, f"Face {i+1}", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Add total count
    cv2.putText(image, f"Faces: {len(faces)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save result
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"{Path(image_path).stem}_landmarks.jpg")
    cv2.imwrite(output_path, image)
    print(f"Output saved: {output_path}")


def run_webcam(detector, landmarker):
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

            # Process each face
            for face in faces:
                # Draw bounding box
                bbox = face['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Get and draw 106 landmarks
                landmarks = landmarker.get_landmarks(frame, bbox)
                for x, y in landmarks.astype(int):
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # Add info
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("106-Point Landmarks", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames")


def main():
    parser = argparse.ArgumentParser(description="Run facial landmark detection")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--webcam", action="store_true", help="Use webcam instead of image")
    parser.add_argument("--detector", type=str, default="retinaface",
                       choices=['retinaface', 'scrfd'], help="Face detector to use")
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

    print("Initializing landmark detector...")
    landmarker = Landmark106()
    print("Models initialized\n")

    # Process
    if args.webcam:
        run_webcam(detector, landmarker)
    else:
        process_image(detector, landmarker, args.image, args.save_dir)


if __name__ == "__main__":
    main()
