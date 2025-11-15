import os
import cv2
import time
import argparse
import numpy as np

from uniface.detection import RetinaFace, SCRFD
from uniface.visualization import draw_detections


def run_inference(detector, image_path: str, vis_threshold: float = 0.6, save_dir: str = "outputs"):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from '{image_path}'")
        return

    # 1. Get the list of face dictionaries from the detector
    faces = detector.detect(image)

    if faces:
        # 2. Unpack the data into separate lists
        bboxes = [face['bbox'] for face in faces]
        scores = [face['confidence'] for face in faces]
        landmarks = [face['landmarks'] for face in faces]

        # 3. Pass the unpacked lists to the drawing function
        draw_detections(image, bboxes, scores, landmarks, vis_threshold=0.6)


    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_out.jpg")
    cv2.imwrite(output_path, image)
    print(f"Output saved at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run face detection on an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument(
        "--method",
        type=str,
        default="retinaface",
        choices=['retinaface', 'scrfd'],
        help="Detection method to use."
    )
    parser.add_argument("--threshold", type=float, default=0.6, help="Visualization confidence threshold")
    parser.add_argument("--iterations", type=int, default=1, help="Number of inference runs for benchmarking")
    parser.add_argument("--save_dir", type=str, default="outputs", help="Directory to save output images")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        from uniface import enable_logging
        enable_logging()

    print(f"Initializing detector: {args.method}")
    if args.method == 'retinaface':
        detector = RetinaFace()
    else:
        detector = SCRFD()

    avg_time = 0
    for i in range(args.iterations):
        start = time.time()
        run_inference(detector, args.image, args.threshold, args.save_dir)
        elapsed = time.time() - start
        print(f"[{i + 1}/{args.iterations}] Inference time: {elapsed:.4f} seconds")
        if i >= 0:  # Avoid counting the first run if it includes model loading time
            avg_time += elapsed

    if args.iterations > 1:
        # Adjust average calculation to exclude potential first-run overhead
        effective_iterations = max(1, args.iterations)
        print(
            f"\nAverage inference time over {effective_iterations} runs: {avg_time / effective_iterations:.4f} seconds")


if __name__ == "__main__":
    main()
