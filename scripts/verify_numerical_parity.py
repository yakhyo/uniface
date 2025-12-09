#!/usr/bin/env python3
"""
Verify numerical parity between MLX and ONNX for RetinaFace.

This script compares the final outputs (cls, bbox, landmarks) between
MLX and ONNX implementations to ensure they match within tolerance.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import mlx.core as mx

from uniface.constants import RetinaFaceWeights
from uniface.detection.retinaface_mlx import RetinaFaceNetworkFused
from uniface.mlx_utils import load_mlx_fused_weights, synchronize
from uniface.model_store import verify_model_weights
from uniface.onnx_utils import create_onnx_session


def compare_arrays(name, mlx_arr, onnx_arr, detailed=False):
    """Compare two arrays and print statistics."""
    if mlx_arr.shape != onnx_arr.shape:
        print(f"  {name}: SHAPE MISMATCH - MLX {mlx_arr.shape} vs ONNX {onnx_arr.shape}")
        return False

    mlx_flat = mlx_arr.flatten()
    onnx_flat = onnx_arr.flatten()

    max_diff = np.abs(mlx_arr - onnx_arr).max()
    mean_diff = np.abs(mlx_arr - onnx_arr).mean()

    if np.std(mlx_flat) > 1e-10 and np.std(onnx_flat) > 1e-10:
        corr = np.corrcoef(mlx_flat, onnx_flat)[0, 1]
    else:
        corr = 0.0

    match = "✓" if corr > 0.999 else "✗"
    print(f"  {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, corr={corr:.6f} {match}")

    if detailed:
        print(f"    MLX range: [{mlx_arr.min():.4f}, {mlx_arr.max():.4f}]")
        print(f"    ONNX range: [{onnx_arr.min():.4f}, {onnx_arr.max():.4f}]")

    return corr > 0.999


def main():
    print("=" * 70)
    print("Numerical Parity Verification: MLX vs ONNX RetinaFace")
    print("=" * 70)

    # Load test image
    test_image_path = Path(__file__).parent.parent / "assets" / "test.jpg"
    if not test_image_path.exists():
        test_image_path = Path("/tmp/test_face.jpg")
    if not test_image_path.exists():
        print("Creating synthetic input...")
        np.random.seed(42)
        image = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    else:
        image = cv2.imread(str(test_image_path))
        image = cv2.resize(image, (640, 640))

    print(f"Input image shape: {image.shape}")

    # Preprocess for ONNX (NCHW format)
    onnx_input = np.float32(image) - np.array([104, 117, 123], dtype=np.float32)
    onnx_input = onnx_input.transpose(2, 0, 1)
    onnx_input = np.expand_dims(onnx_input, 0)

    # Preprocess for MLX (NHWC format)
    mlx_input = np.float32(image) - np.array([104, 117, 123], dtype=np.float32)
    mlx_input = np.expand_dims(mlx_input, 0)
    mlx_tensor = mx.array(mlx_input)

    # Load ONNX model
    print("\nLoading ONNX model...")
    onnx_path = verify_model_weights(RetinaFaceWeights.MNET_V2)
    onnx_session = create_onnx_session(onnx_path)

    # Run ONNX inference
    print("Running ONNX inference...")
    input_name = onnx_session.get_inputs()[0].name
    onnx_outputs = onnx_session.run(None, {input_name: onnx_input})
    onnx_bbox = onnx_outputs[0]
    onnx_cls = onnx_outputs[1]
    onnx_landmarks = onnx_outputs[2]

    # Load MLX model
    print("Loading MLX model...")
    mlx_model = RetinaFaceNetworkFused(backbone_type='mobilenetv2', width_mult=1.0)
    weights_path = Path(__file__).parent.parent / "weights_mlx_fused" / "retinaface_mnet_v2.safetensors"

    if weights_path.exists():
        load_mlx_fused_weights(mlx_model, str(weights_path))
    else:
        print(f"ERROR: Fused weights not found at {weights_path}")
        return

    mlx_model.train(False)

    # Run MLX inference
    print("Running MLX inference...")
    cls_preds, bbox_preds, landmark_preds = mlx_model(mlx_tensor)
    synchronize(cls_preds, bbox_preds, landmark_preds)

    mlx_cls = np.array(cls_preds)
    mlx_bbox = np.array(bbox_preds)
    mlx_landmarks = np.array(landmark_preds)

    # Apply softmax to MLX cls (ONNX already has softmax applied)
    mlx_cls_softmax = np.exp(mlx_cls) / np.exp(mlx_cls).sum(axis=-1, keepdims=True)

    print("\n" + "=" * 70)
    print("Final Output Comparison")
    print("=" * 70)

    print(f"\nOutput shapes:")
    print(f"  bbox: MLX {mlx_bbox.shape}, ONNX {onnx_bbox.shape}")
    print(f"  cls: MLX {mlx_cls_softmax.shape}, ONNX {onnx_cls.shape}")
    print(f"  landmarks: MLX {mlx_landmarks.shape}, ONNX {onnx_landmarks.shape}")

    print("\nOverall comparison:")
    cls_match = compare_arrays("Classification", mlx_cls_softmax, onnx_cls, detailed=True)
    bbox_match = compare_arrays("BBox regression", mlx_bbox, onnx_bbox, detailed=True)
    lmk_match = compare_arrays("Landmarks", mlx_landmarks, onnx_landmarks, detailed=True)

    # Per-level comparison
    print("\n" + "=" * 70)
    print("Per-Level Comparison")
    print("=" * 70)

    level_sizes = [80*80*2, 40*40*2, 20*20*2]
    level_names = ['P1 (80x80, stride 8)', 'P2 (40x40, stride 16)', 'P3 (20x20, stride 32)']

    all_match = True
    start = 0
    for level_name, size in zip(level_names, level_sizes):
        end = start + size
        print(f"\n{level_name}:")
        cls_ok = compare_arrays("  cls", mlx_cls_softmax[:, start:end, :], onnx_cls[:, start:end, :])
        bbox_ok = compare_arrays("  bbox", mlx_bbox[:, start:end, :], onnx_bbox[:, start:end, :])
        lmk_ok = compare_arrays("  landmarks", mlx_landmarks[:, start:end, :], onnx_landmarks[:, start:end, :])
        all_match = all_match and cls_ok and bbox_ok and lmk_ok
        start = end

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if all_match:
        print("\n✓ SUCCESS: MLX and ONNX outputs match within tolerance!")
        print("  All outputs have correlation > 0.999")
    else:
        print("\n✗ FAILURE: Some outputs do not match")

    # Show sample predictions
    print("\n" + "=" * 70)
    print("Sample Predictions (first 5 anchors)")
    print("=" * 70)

    print("\nBBox (first 5):")
    print(f"  ONNX: {onnx_bbox[0, :5, :]}")
    print(f"  MLX:  {mlx_bbox[0, :5, :]}")

    print("\nConfidence (first 5, face class):")
    print(f"  ONNX: {onnx_cls[0, :5, 1]}")
    print(f"  MLX:  {mlx_cls_softmax[0, :5, 1]}")


if __name__ == "__main__":
    main()
