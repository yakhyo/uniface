# Face detection on video files
# Usage: python run_video_detection.py --input video.mp4 --output output.mp4

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from uniface import SCRFD, RetinaFace
from uniface.visualization import draw_detections


def process_video(detector, input_path: str, output_path: str, threshold: float = 0.6, show_preview: bool = False):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{input_path}'")
        return

    # get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Input: {input_path} ({width}x{height}, {fps:.1f} fps, {total_frames} frames)")
    print(f"Output: {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec for .mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Cannot create output video '{output_path}'")
        cap.release()
        return

    frame_count = 0
    total_faces = 0

    for _ in tqdm(range(total_frames), desc="Processing", unit="frames"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        faces = detector.detect(frame)
        total_faces += len(faces)

        bboxes = [f["bbox"] for f in faces]
        scores = [f["confidence"] for f in faces]
        landmarks = [f["landmarks"] for f in faces]
        draw_detections(frame, bboxes, scores, landmarks, vis_threshold=threshold)

        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)

        if show_preview:
            cv2.imshow("Processing - Press 'q' to cancel", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nCancelled by user")
                break

    cap.release()
    out.release()
    if show_preview:
        cv2.destroyAllWindows()

    avg_faces = total_faces / frame_count if frame_count > 0 else 0
    print(f"\nDone! {frame_count} frames, {total_faces} faces ({avg_faces:.1f} avg/frame)")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Process video with face detection")
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, required=True, help="Output video path")
    parser.add_argument("--detector", type=str, default="retinaface", choices=["retinaface", "scrfd"])
    parser.add_argument("--threshold", type=float, default=0.6, help="Visualization threshold")
    parser.add_argument("--preview", action="store_true", help="Show live preview")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' does not exist")
        return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    detector = RetinaFace() if args.detector == "retinaface" else SCRFD()
    process_video(detector, args.input, args.output, args.threshold, args.preview)


if __name__ == "__main__":
    main()
