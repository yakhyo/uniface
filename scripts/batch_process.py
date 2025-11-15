"""Batch Image Processing Script"""

import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

from uniface import RetinaFace, SCRFD
from uniface.visualization import draw_detections


def get_image_files(input_dir: Path, extensions: tuple) -> list:
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(f"*.{ext}"))
        image_files.extend(input_dir.glob(f"*.{ext.upper()}"))

    return sorted(image_files)


def process_single_image(detector, image_path: Path, output_dir: Path,
                        vis_threshold: float, skip_existing: bool) -> dict:
    output_path = output_dir / f"{image_path.stem}_detected{image_path.suffix}"

    # Skip if already processed
    if skip_existing and output_path.exists():
        return {"status": "skipped", "faces": 0}

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        return {"status": "error", "error": "Failed to load image"}

    # Detect faces
    try:
        faces = detector.detect(image)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # Draw detections
    bboxes = [f['bbox'] for f in faces]
    scores = [f['confidence'] for f in faces]
    landmarks = [f['landmarks'] for f in faces]
    draw_detections(image, bboxes, scores, landmarks, vis_threshold=vis_threshold)

    # Add face count
    cv2.putText(image, f"Faces: {len(faces)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save result
    cv2.imwrite(str(output_path), image)

    return {"status": "success", "faces": len(faces)}


def batch_process(detector, input_dir: str, output_dir: str, extensions: tuple,
                 vis_threshold: float, skip_existing: bool):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Get image files
    image_files = get_image_files(input_path, extensions)

    if not image_files:
        print(f"No image files found in '{input_dir}' with extensions {extensions}")
        return

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Found {len(image_files)} images\n")

    # Process images
    results = {
        "success": 0,
        "skipped": 0,
        "error": 0,
        "total_faces": 0
    }

    with tqdm(image_files, desc="Processing images", unit="img") as pbar:
        for image_path in pbar:
            result = process_single_image(
                detector, image_path, output_path,
                vis_threshold, skip_existing
            )

            if result["status"] == "success":
                results["success"] += 1
                results["total_faces"] += result["faces"]
                pbar.set_postfix({"faces": result["faces"]})
            elif result["status"] == "skipped":
                results["skipped"] += 1
            else:
                results["error"] += 1
                print(f"\nError processing {image_path.name}: {result.get('error', 'Unknown error')}")

    # Print summary
    print(f"\nBatch processing complete!")
    print(f"   Total images: {len(image_files)}")
    print(f"   Successfully processed: {results['success']}")
    print(f"   Skipped: {results['skipped']}")
    print(f"   Errors: {results['error']}")
    print(f"   Total faces detected: {results['total_faces']}")
    if results['success'] > 0:
        print(f"   Average faces per image: {results['total_faces']/results['success']:.2f}")
    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Batch process images with face detection")
    parser.add_argument("--input", type=str, required=True,
                       help="Input directory containing images")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory for processed images")
    parser.add_argument("--detector", type=str, default="retinaface",
                       choices=['retinaface', 'scrfd'], help="Face detector to use")
    parser.add_argument("--threshold", type=float, default=0.6,
                       help="Confidence threshold for visualization")
    parser.add_argument("--extensions", type=str, default="jpg,jpeg,png,bmp",
                       help="Comma-separated list of image extensions")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip files that already exist in output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Check input directory exists
    if not Path(args.input).exists():
        print(f"Error: Input directory '{args.input}' does not exist")
        return

    if args.verbose:
        from uniface import enable_logging
        enable_logging()

    # Parse extensions
    extensions = tuple(ext.strip() for ext in args.extensions.split(','))

    # Initialize detector
    print(f"Initializing detector: {args.detector}")
    if args.detector == 'retinaface':
        detector = RetinaFace()
    else:
        detector = SCRFD()
    print("Detector initialized\n")

    # Process batch
    batch_process(detector, args.input, args.output, extensions,
                 args.threshold, args.skip_existing)


if __name__ == "__main__":
    main()
