import os
import cv2
import time
import argparse
import numpy as np

from uniface.detection import RetinaFace, draw_detections, SCRFD
from uniface.constants import RetinaFaceWeights, SCRFDWeights


def run_inference(model, image_path, vis_threshold=0.6, save_dir="outputs"):
    """
    Run face detection on a single image.

    Args:
        model (RetinaFace): Initialized RetinaFace model.
        image_path (str): Path to input image.
        vis_threshold (float): Threshold for drawing detections.
        save_dir (str): Directory to save output image.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Failed to load image from '{image_path}'")
        return

    boxes, landmarks = model.detect(image)
    draw_detections(image, (boxes, landmarks), vis_threshold)

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_out.jpg")
    cv2.imwrite(output_path, image)
    print(f"✅ Output saved at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run RetinaFace inference on an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--model", type=str, default="MNET_V2", choices=[m.name for m in RetinaFaceWeights], help="Model variant to use")
    parser.add_argument("--threshold", type=float, default=0.6, help="Visualization confidence threshold")
    parser.add_argument("--iterations", type=int, default=1, help="Number of inference runs for benchmarking")
    parser.add_argument("--save_dir", type=str, default="outputs", help="Directory to save output images")

    args = parser.parse_args()

    model_name = RetinaFaceWeights[args.model]
    model = RetinaFace(model_name=model_name)

    avg_time = 0
    for i in range(args.iterations):
        start = time.time()
        run_inference(model, args.image, args.threshold, args.save_dir)
        elapsed = time.time() - start
        print(f"[{i + 1}/{args.iterations}] ⏱️ Inference time: {elapsed:.4f} seconds")
        avg_time += elapsed

    if args.iterations > 1:
        print(f"\n🔥 Average inference time over {args.iterations} runs: {avg_time / args.iterations:.4f} seconds")


if __name__ == "__main__":
    main()
