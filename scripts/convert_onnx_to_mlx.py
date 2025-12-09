#!/usr/bin/env python3
# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo
#
# ONNX to MLX Weight Conversion Script
# Extracts fused weights from ONNX model for MLX inference

"""
Convert ONNX model weights to MLX-compatible safetensors format.

This script:
1. Loads ONNX model with fused BatchNorm (BN folded into Conv)
2. Traces the graph to map weight names to model components
3. Transposes Conv2d weights from OIHW to OHWI format
4. Saves as .safetensors

The key difference from PyTorch conversion is that ONNX has fused weights,
meaning Conv layers have bias and BatchNorm statistics are already applied.

Usage:
    python scripts/convert_onnx_to_mlx.py --model retinaface_mnet_v2
"""

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    print('ONNX is required for weight conversion. Install with: pip install onnx')
    sys.exit(1)

try:
    from safetensors.numpy import save_file
except ImportError:
    print('safetensors is required. Install with: pip install safetensors')
    sys.exit(1)

from uniface.constants import RetinaFaceWeights
from uniface.model_store import verify_model_weights


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def transpose_conv_weight(weight: np.ndarray) -> np.ndarray:
    """
    Transpose Conv2d weight from ONNX OIHW to MLX OHWI format.

    ONNX/PyTorch: (out_channels, in_channels, height, width)
    MLX: (out_channels, height, width, in_channels)
    """
    if weight.ndim == 4:
        return weight.transpose(0, 2, 3, 1)
    return weight


class ONNXGraphTracer:
    """
    Traces ONNX graph to map weight names to model components.

    The ONNX model has fused BatchNorm, so Conv nodes have both weight and bias.
    We need to trace the graph to understand which weights belong to which
    component (backbone, FPN, SSH, heads).
    """

    def __init__(self, model: onnx.ModelProto):
        self.model = model
        self.initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

        # Build node input/output maps
        self.node_by_output = {}
        for node in model.graph.node:
            for output in node.output:
                self.node_by_output[output] = node

        # Build weight to node map
        self.weight_to_node = {}
        for node in model.graph.node:
            if node.op_type == 'Conv':
                for i, inp in enumerate(node.input):
                    if inp in self.initializers:
                        self.weight_to_node[inp] = (node, i)

    def get_weight_mapping(self) -> Dict[str, str]:
        """
        Map ONNX weight names to MLX model parameter names.

        Returns:
            Dict mapping ONNX weight name -> MLX parameter key
        """
        mapping = {}

        # Process each Conv node and map its weights
        for node in self.model.graph.node:
            if node.op_type != 'Conv':
                continue

            mlx_prefix = self._get_mlx_prefix_for_conv(node)
            if mlx_prefix is None:
                continue

            # Map weight and bias
            for i, inp in enumerate(node.input[1:], 1):  # Skip input tensor
                if inp in self.initializers:
                    if i == 1:
                        mapping[inp] = f'{mlx_prefix}.weight'
                    elif i == 2:
                        mapping[inp] = f'{mlx_prefix}.bias'

        # Handle detection heads (they have proper names in ONNX)
        for init in self.model.graph.initializer:
            name = init.name
            if 'class_head' in name or 'bbox_head' in name or 'landmark_head' in name:
                # These are already properly named, just add .layers. for MLX Sequential
                parts = name.split('.')
                if len(parts) == 4:  # e.g., class_head.class_head.0.weight
                    head_type = parts[0]
                    idx = parts[2]
                    param = parts[3]
                    mapping[name] = f'{head_type}.{head_type}.layers.{idx}.{param}'

        return mapping

    def _get_mlx_prefix_for_conv(self, node: onnx.NodeProto) -> Optional[str]:
        """Get MLX parameter prefix for a Conv node based on its name."""
        name = node.name

        # Stem: /fx/features.0/features.0.0/Conv
        if '/fx/features.0/features.0.0/Conv' in name:
            return 'backbone.stem.conv'

        # Inverted residuals: /fx/features.N/conv/...
        if '/fx/features.' in name and '/conv/' in name:
            return self._map_inverted_residual_conv(name)

        # Final 1x1 conv: /fx/features.18/features.18.0/Conv
        if '/fx/features.18/features.18.0/Conv' in name:
            return 'backbone.final_conv.conv'

        # FPN output layers: /fpn/output1/output1.0/Conv
        if '/fpn/output' in name:
            match = self._extract_fpn_output(name)
            if match:
                return match

        # FPN merge layers: /fpn/merge1/merge1.0/Conv
        if '/fpn/merge' in name:
            match = self._extract_fpn_merge(name)
            if match:
                return match

        # SSH layers: /ssh1/conv3X3/conv3X3.0/Conv
        if '/ssh' in name:
            return self._map_ssh_conv(name)

        return None

    def _map_inverted_residual_conv(self, name: str) -> Optional[str]:
        """Map inverted residual Conv node to MLX key."""
        # ONNX node name patterns:
        # - Stage 1 (t=1): /fx/features.1/conv/conv.0/conv.0.0/Conv (dw)
        #                  /fx/features.1/conv/conv.1/Conv (project)
        # - Stages 2-7 (t=6): /fx/features.N/conv/conv.0/conv.0.0/Conv (expand)
        #                     /fx/features.N/conv/conv.1/conv.1.0/Conv (dw)
        #                     /fx/features.N/conv/conv.2/Conv (project)

        # Try pattern with sub-index first: conv.X/conv.X.Y/Conv
        match = re.search(r'/fx/features\.(\d+)/conv/conv\.(\d+)/conv\.\d+\.(\d+)/Conv', name)
        if match:
            features_idx = int(match.group(1))
            conv_idx = int(match.group(2))
            sub_idx = int(match.group(3))
        else:
            # Try pattern without sub-index: conv.X/Conv
            match = re.search(r'/fx/features\.(\d+)/conv/conv\.(\d+)/Conv', name)
            if match:
                features_idx = int(match.group(1))
                conv_idx = int(match.group(2))
                sub_idx = None
            else:
                return None

        # Map features index to stage/layer
        stage_mapping = [
            (1, 1, 1),  # features 1: stage 1, 1 block
            (2, 2, 2),  # features 2-3: stage 2, 2 blocks
            (4, 3, 3),  # features 4-6: stage 3, 3 blocks
            (7, 4, 4),  # features 7-10: stage 4, 4 blocks
            (11, 5, 3),  # features 11-13: stage 5, 3 blocks
            (14, 6, 3),  # features 14-16: stage 6, 3 blocks
            (17, 7, 1),  # features 17: stage 7, 1 block
        ]

        stage = None
        layer = None
        for start_idx, stage_num, num_blocks in stage_mapping:
            if start_idx <= features_idx < start_idx + num_blocks:
                stage = stage_num
                layer = features_idx - start_idx
                break

        if stage is None:
            return None

        # Get expand_ratio for this stage
        expand_ratios = [1, 6, 6, 6, 6, 6, 6]  # stage 1-7
        t = expand_ratios[stage - 1]

        if t == 1:
            # No expansion: conv.0/conv.0.0 = dw, conv.1 = project
            if conv_idx == 0 and sub_idx == 0:
                return f'backbone.stage{stage}.layers.{layer}.depthwise'
            elif conv_idx == 1 and sub_idx is None:
                return f'backbone.stage{stage}.layers.{layer}.project'
        else:
            # With expansion: conv.0/conv.0.0 = expand, conv.1/conv.1.0 = dw, conv.2 = project
            if conv_idx == 0 and sub_idx == 0:
                return f'backbone.stage{stage}.layers.{layer}.expand.layers.0'
            elif conv_idx == 1 and sub_idx == 0:
                return f'backbone.stage{stage}.layers.{layer}.depthwise'
            elif conv_idx == 2 and sub_idx is None:
                return f'backbone.stage{stage}.layers.{layer}.project'

        return None

    def _extract_fpn_output(self, name: str) -> Optional[str]:
        """Extract FPN output layer key."""
        match = re.search(r'/fpn/output(\d+)/output\d+\.0/Conv', name)
        if match:
            idx = match.group(1)
            return f'fpn.output{idx}.conv'
        return None

    def _extract_fpn_merge(self, name: str) -> Optional[str]:
        """Extract FPN merge layer key."""
        match = re.search(r'/fpn/merge(\d+)/merge\d+\.0/Conv', name)
        if match:
            idx = match.group(1)
            return f'fpn.merge{idx}.conv'
        return None

    def _map_ssh_conv(self, name: str) -> Optional[str]:
        """Map SSH Conv node to MLX key."""
        # Pattern: /ssh1/conv3X3/conv3X3.0/Conv
        match = re.search(r'/ssh(\d+)/(\w+)/\w+\.0/Conv', name)
        if match:
            ssh_idx = match.group(1)
            conv_name = match.group(2)
            return f'ssh{ssh_idx}.{conv_name}.conv'
        return None


def convert_retinaface_mnetv2_from_onnx(
    output_path: str,
    verbose: bool = True,
) -> str:
    """
    Convert RetinaFace MobileNetV2 weights from ONNX to MLX format.

    Args:
        output_path: Path for output .safetensors file
        verbose: Print conversion details

    Returns:
        SHA-256 hash of the output file
    """
    # Load ONNX model
    onnx_path = verify_model_weights(RetinaFaceWeights.MNET_V2)
    print(f'\nLoading ONNX model: {onnx_path}')

    model = onnx.load(onnx_path)

    # Create graph tracer
    tracer = ONNXGraphTracer(model)

    # Get weight mapping
    weight_mapping = tracer.get_weight_mapping()

    if verbose:
        print(f'\nMapped {len(weight_mapping)} weights')

    # Convert weights
    mlx_weights = {}
    transposed_count = 0

    for onnx_name, mlx_name in weight_mapping.items():
        weight = tracer.initializers[onnx_name]

        # Transpose conv weights
        if 'weight' in mlx_name and weight.ndim == 4:
            original_shape = weight.shape
            weight = transpose_conv_weight(weight)
            transposed_count += 1
            if verbose:
                print(f'  [TRANSPOSE] {onnx_name} -> {mlx_name}: {original_shape} -> {weight.shape}')
        elif verbose:
            print(f'  [MAP] {onnx_name} -> {mlx_name}: {weight.shape}')

        mlx_weights[mlx_name] = weight

    print('\nConversion summary:')
    print(f'  Mapped: {len(mlx_weights)}')
    print(f'  Transposed: {transposed_count}')

    # Save as safetensors
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    save_file(mlx_weights, output_path)
    print(f'\nSaved to: {output_path}')

    # Compute hash
    sha256 = compute_sha256(output_path)
    print(f'SHA-256: {sha256}')

    return sha256


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX weights to MLX format')
    parser.add_argument('--model', '-m', type=str, default='retinaface_mnet_v2', help='Model to convert')
    parser.add_argument('--output', '-o', type=str, help='Output .safetensors file')
    parser.add_argument('--output-dir', type=str, default='weights_mlx_fused', help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.model == 'retinaface_mnet_v2':
        output_path = args.output or os.path.join(args.output_dir, 'retinaface_mnet_v2.safetensors')
        convert_retinaface_mnetv2_from_onnx(output_path, args.verbose)
    else:
        parser.print_help()
        print('\nSupported models:')
        print('  retinaface_mnet_v2')


if __name__ == '__main__':
    main()
