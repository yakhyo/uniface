#!/usr/bin/env python3
"""
Debug script to compare FPN top-down pathway between MLX and ONNX.

Key finding: All laterals match perfectly (corr = 1.0).
Issue is in the top-down pathway (upsample + add).

ONNX Add outputs:
- fpn/Add (P2 after add): upsample(P3) + P2 lateral -> range [0, 8.8878]
- fpn/Add_1 (P1 after add): upsample(P2 after merge?) + P1 lateral -> range [0, 9.3575]

Wait - ONNX might be adding the MERGED P2 to P1, not the raw P2 after add!
Let's check the ONNX graph order.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import onnx
from onnx import numpy_helper
import mlx.core as mx

from uniface.constants import RetinaFaceWeights
from uniface.detection.retinaface_mlx import RetinaFaceNetworkFused
from uniface.mlx_utils import load_mlx_fused_weights, synchronize
from uniface.model_store import verify_model_weights


def get_onnx_intermediate(onnx_path, input_data, output_names):
    """Get intermediate ONNX tensors by modifying the graph outputs."""
    import onnxruntime as ort

    model = onnx.load(onnx_path)

    for name in output_names:
        existing_names = [o.name for o in model.graph.output]
        if name not in existing_names:
            model.graph.output.append(
                onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
            )

    session = ort.InferenceSession(
        model.SerializeToString(),
        providers=['CoreMLExecutionProvider', 'CPUExecutionProvider']
    )

    input_name = session.get_inputs()[0].name
    results = session.run(output_names, {input_name: input_data})

    return dict(zip(output_names, results))


def trace_onnx_fpn_order(onnx_path):
    """Trace the exact order of FPN operations in ONNX."""
    model = onnx.load(onnx_path)

    print("\n" + "=" * 70)
    print("ONNX FPN Operation Order")
    print("=" * 70)

    # Find all FPN nodes in order
    fpn_nodes = []
    for node in model.graph.node:
        if '/fpn/' in node.name:
            fpn_nodes.append(node)

    # Print key operations
    for node in fpn_nodes:
        if node.op_type in ['Conv', 'Add', 'Resize', 'LeakyRelu']:
            inputs_str = ', '.join(node.input[:3])
            outputs_str = ', '.join(node.output)
            print(f"  {node.op_type}: {node.name[:60]}")
            print(f"    inputs: {inputs_str}")
            print(f"    output: {outputs_str}")

    return model


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

    match = "✓" if corr > 0.99 else "✗"
    print(f"  {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, corr={corr:.6f} {match}")

    if detailed:
        print(f"    MLX range: [{mlx_arr.min():.4f}, {mlx_arr.max():.4f}]")
        print(f"    ONNX range: [{onnx_arr.min():.4f}, {onnx_arr.max():.4f}]")

    return corr > 0.99


def main():
    print("=" * 70)
    print("FPN Add Pathway Debug: MLX vs ONNX")
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

    # Load ONNX model path
    onnx_path = verify_model_weights(RetinaFaceWeights.MNET_V2)

    # Trace ONNX FPN order to understand the computation graph
    onnx_model = trace_onnx_fpn_order(onnx_path)

    # Preprocess for ONNX (NCHW format)
    onnx_input = np.float32(image) - np.array([104, 117, 123], dtype=np.float32)
    onnx_input = onnx_input.transpose(2, 0, 1)
    onnx_input = np.expand_dims(onnx_input, 0)

    # Preprocess for MLX (NHWC format)
    mlx_input = np.float32(image) - np.array([104, 117, 123], dtype=np.float32)
    mlx_input = np.expand_dims(mlx_input, 0)
    mlx_tensor = mx.array(mlx_input)

    # Get ONNX add pathway outputs
    print("\n" + "=" * 70)
    print("Getting ONNX Add Pathway Outputs")
    print("=" * 70)

    intermediate_names = [
        "/fpn/output1/output1.2/LeakyRelu_output_0",  # P1 lateral
        "/fpn/output2/output2.2/LeakyRelu_output_0",  # P2 lateral
        "/fpn/output3/output3.2/LeakyRelu_output_0",  # P3 lateral
        "/fpn/Resize_output_0",  # P3 upsampled to 40x40
        "/fpn/Add_output_0",  # P2 after add (P2_lat + upsample(P3))
        "/fpn/merge2/merge2.2/LeakyRelu_output_0",  # P2 merged
        "/fpn/Resize_1_output_0",  # P2_merged upsampled to 80x80
        "/fpn/Add_1_output_0",  # P1 after add
        "/fpn/merge1/merge1.2/LeakyRelu_output_0",  # P1 merged (final P1)
    ]

    onnx_outputs = get_onnx_intermediate(onnx_path, onnx_input, intermediate_names)

    print("\nONNX intermediate outputs:")
    for name, arr in onnx_outputs.items():
        short_name = name.split('/')[-1]
        print(f"  {short_name}: {arr.shape}, range=[{arr.min():.4f}, {arr.max():.4f}]")

    # Load MLX model
    print("\n" + "=" * 70)
    print("Running MLX FPN Step by Step")
    print("=" * 70)

    mlx_model = RetinaFaceNetworkFused(backbone_type='mobilenetv2', width_mult=1.0)
    weights_path = Path(__file__).parent.parent / "weights_mlx_fused" / "retinaface_mnet_v2.safetensors"

    if weights_path.exists():
        load_mlx_fused_weights(mlx_model, str(weights_path))
    else:
        print(f"ERROR: Fused weights not found at {weights_path}")
        return

    mlx_model.train(False)

    # Run backbone
    features = mlx_model.backbone(mlx_tensor)
    synchronize(*features)

    c1, c2, c3 = features
    fpn = mlx_model.fpn

    # Step 1: Lateral connections
    p3_lat = fpn.output3(c3)
    p2_lat = fpn.output2(c2)
    p1_lat = fpn.output1(c1)
    synchronize(p3_lat, p2_lat, p1_lat)

    # Step 2: First upsample + add (P2)
    h2, w2 = p2_lat.shape[1:3]
    p3_up = fpn._upsample(p3_lat, (h2, w2))
    synchronize(p3_up)

    p2_add = p2_lat + p3_up
    synchronize(p2_add)

    # Step 3: P2 merge (smoothing)
    p2_merged = fpn.merge2(p2_add)
    synchronize(p2_merged)

    # Step 4: Second upsample + add (P1)
    # KEY QUESTION: Does ONNX upsample p2_add or p2_merged?
    h1, w1 = p1_lat.shape[1:3]

    # Option A: upsample p2_add (current MLX implementation)
    p2_add_up = fpn._upsample(p2_add, (h1, w1))
    synchronize(p2_add_up)
    p1_add_from_p2_add = p1_lat + p2_add_up
    synchronize(p1_add_from_p2_add)

    # Option B: upsample p2_merged (might be what ONNX does)
    p2_merged_up = fpn._upsample(p2_merged, (h1, w1))
    synchronize(p2_merged_up)
    p1_add_from_p2_merged = p1_lat + p2_merged_up
    synchronize(p1_add_from_p2_merged)

    # Step 5: P1 merge
    p1_merged_a = fpn.merge1(p1_add_from_p2_add)
    p1_merged_b = fpn.merge1(p1_add_from_p2_merged)
    synchronize(p1_merged_a, p1_merged_b)

    # Compare with ONNX
    print("\n" + "=" * 70)
    print("Comparison: MLX vs ONNX")
    print("=" * 70)

    onnx_p1_lat = onnx_outputs["/fpn/output1/output1.2/LeakyRelu_output_0"].transpose(0, 2, 3, 1)
    onnx_p2_lat = onnx_outputs["/fpn/output2/output2.2/LeakyRelu_output_0"].transpose(0, 2, 3, 1)
    onnx_p3_lat = onnx_outputs["/fpn/output3/output3.2/LeakyRelu_output_0"].transpose(0, 2, 3, 1)
    onnx_p3_up = onnx_outputs["/fpn/Resize_output_0"].transpose(0, 2, 3, 1)
    onnx_p2_add = onnx_outputs["/fpn/Add_output_0"].transpose(0, 2, 3, 1)
    onnx_p2_merged = onnx_outputs["/fpn/merge2/merge2.2/LeakyRelu_output_0"].transpose(0, 2, 3, 1)
    onnx_p2_up = onnx_outputs["/fpn/Resize_1_output_0"].transpose(0, 2, 3, 1)
    onnx_p1_add = onnx_outputs["/fpn/Add_1_output_0"].transpose(0, 2, 3, 1)
    onnx_p1_merged = onnx_outputs["/fpn/merge1/merge1.2/LeakyRelu_output_0"].transpose(0, 2, 3, 1)

    print("\n1. Lateral connections:")
    compare_arrays("P1 lateral", np.array(p1_lat), onnx_p1_lat)
    compare_arrays("P2 lateral", np.array(p2_lat), onnx_p2_lat)
    compare_arrays("P3 lateral", np.array(p3_lat), onnx_p3_lat)

    print("\n2. P3 upsampled (to P2 size):")
    compare_arrays("P3 upsampled", np.array(p3_up), onnx_p3_up, detailed=True)

    print("\n3. P2 after add:")
    compare_arrays("P2 add", np.array(p2_add), onnx_p2_add, detailed=True)

    print("\n4. P2 merged:")
    compare_arrays("P2 merged", np.array(p2_merged), onnx_p2_merged, detailed=True)

    print("\n5. What gets upsampled to P1?")
    print(f"   ONNX Resize_1 shape: {onnx_p2_up.shape}, range=[{onnx_p2_up.min():.4f}, {onnx_p2_up.max():.4f}]")
    print(f"   MLX p2_add_up shape: {np.array(p2_add_up).shape}, range=[{np.array(p2_add_up).min():.4f}, {np.array(p2_add_up).max():.4f}]")
    print(f"   MLX p2_merged_up shape: {np.array(p2_merged_up).shape}, range=[{np.array(p2_merged_up).min():.4f}, {np.array(p2_merged_up).max():.4f}]")

    # Check which one matches
    compare_arrays("ONNX_p2_up vs MLX_p2_add_up", np.array(p2_add_up), onnx_p2_up, detailed=True)
    compare_arrays("ONNX_p2_up vs MLX_p2_merged_up", np.array(p2_merged_up), onnx_p2_up, detailed=True)

    print("\n6. P1 after add:")
    print(f"   ONNX P1 add: range=[{onnx_p1_add.min():.4f}, {onnx_p1_add.max():.4f}]")
    compare_arrays("P1 add (from p2_add)", np.array(p1_add_from_p2_add), onnx_p1_add, detailed=True)
    compare_arrays("P1 add (from p2_merged)", np.array(p1_add_from_p2_merged), onnx_p1_add, detailed=True)

    print("\n7. P1 merged (final):")
    compare_arrays("P1 merged (from p2_add)", np.array(p1_merged_a), onnx_p1_merged, detailed=True)
    compare_arrays("P1 merged (from p2_merged)", np.array(p1_merged_b), onnx_p1_merged, detailed=True)


if __name__ == "__main__":
    main()
