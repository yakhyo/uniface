"""Video Face Detection Script"""

import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

from uniface import RetinaFace, SCRFD
from uniface.visualization import draw_detections


def process_video(detector, input_path: str, output_path: str, vis_threshold: float = 0.6,
                 fps: int = None, show_preview: bool = False):
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{input_path}'")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_fps = fps if fps is not None else source_fps

    print(f"📹 Input: {input_path}")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {source_fps:.2f}")
    print(f"   Total frames: {total_frames}")
    print(f"\n📹 Output: {output_path}")
    print(f"   FPS: {output_fps:.2f}\n")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

    if not out.isOpened():
        print(f"Error: Cannot create output video '{output_path}'")
        cap.release()
        return

    # Process frames
    frame_count = 0
    total_faces = 0

    try:
        with tqdm(total=total_frames, desc="Processing", unit="frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Detect faces
                faces = detector.detect(frame)
                total_faces += len(faces)

                # Draw detections
                bboxes = [f['bbox'] for f in faces]
                scores = [f['confidence'] for f in faces]
                landmarks = [f['landmarks'] for f in faces]
                draw_detections(frame, bboxes, scores, landmarks, vis_threshold=vis_threshold)

                # Add frame info
                cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Write frame
                out.write(frame)

                # Show preview if requested
                if show_preview:
                    cv2.imshow("Processing Video - Press 'q' to cancel", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nProcessing cancelled by user")
                        break

                pbar.update(1)

    except KeyboardInterrupt:
        print("\nProcessing interrupted")
    finally:
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()

    # Summary
    print(f"\nProcessing complete!")
    print(f"   Processed: {frame_count} frames")
    print(f"   Total faces detected: {total_faces}")
    print(f"   Average faces per frame: {total_faces/frame_count:.2f}" if frame_count > 0 else "")
    print(f"   Output saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Process video with face detection")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, required=True, help="Path to output video")
    parser.add_argument("--detector", type=str, default="retinaface",
                       choices=['retinaface', 'scrfd'], help="Face detector to use")
    parser.add_argument("--threshold", type=float, default=0.6,
                       help="Confidence threshold for visualization")
    parser.add_argument("--fps", type=int, default=None,
                       help="Output FPS (default: same as input)")
    parser.add_argument("--preview", action="store_true",
                       help="Show live preview during processing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Check input exists
    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' does not exist")
        return

    # Create output directory if needed
    output_dir = Path(args.output).parent
    if output_dir != Path('.'):
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        from uniface import enable_logging
        enable_logging()

    # Initialize detector
    print(f"Initializing detector: {args.detector}")
    if args.detector == 'retinaface':
        detector = RetinaFace()
    else:
        detector = SCRFD()
    print("Detector initialized\n")

    # Process video
    process_video(detector, args.input, args.output, args.threshold, args.fps, args.preview)


if __name__ == "__main__":
    main()
