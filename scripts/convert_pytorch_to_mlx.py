#!/usr/bin/env python3
# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo
#
# PyTorch to MLX Weight Conversion Script with Proper Key Mapping

"""
Convert PyTorch model weights to MLX-compatible safetensors format.

This script:
1. Loads PyTorch .pth weights
2. Maps PyTorch weight keys to MLX model parameter names
3. Transposes Conv2d weights from OIHW to OHWI format
4. Saves as .safetensors

Usage:
    python scripts/convert_pytorch_to_mlx.py --model retinaface_mnet_v2
    python scripts/convert_pytorch_to_mlx.py --all
"""

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
except ImportError:
    print('PyTorch is required for weight conversion. Install with: pip install torch')
    sys.exit(1)

try:
    from safetensors.numpy import save_file
except ImportError:
    print('safetensors is required. Install with: pip install safetensors')
    sys.exit(1)


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def transpose_conv_weight(weight: np.ndarray) -> np.ndarray:
    """
    Transpose Conv2d weight from PyTorch OIHW to MLX OHWI format.

    PyTorch: (out_channels, in_channels, height, width)
    MLX: (out_channels, height, width, in_channels)
    """
    if weight.ndim == 4:
        return weight.transpose(0, 2, 3, 1)
    return weight


class RetinaFaceMNetV2KeyMapper:
    """
    Maps PyTorch RetinaFace MobileNetV2 weight keys to MLX model keys.

    PyTorch structure:
    - fx.features.0.0 = stem conv
    - fx.features.0.1 = stem bn
    - fx.features.1.conv.X.Y = first inverted residual (no expand)
    - fx.features.N.conv.X.Y = inverted residuals with expand
    - fx.features.18.0 = final 1x1 conv (not used in RetinaFace FPN)
    - fpn.* = FPN layers
    - ssh* = SSH context modules
    - *_head.* = detection heads

    MLX structure:
    - backbone.stem.conv/bn = stem
    - backbone.stageN.layers.M.expand/depthwise/project = inverted residuals
    - fpn.* = FPN layers (same naming)
    - ssh* = SSH context modules (same naming)
    - *_head.* = detection heads (same naming)
    """

    # MobileNetV2 configuration: (expand_ratio, out_channels, num_blocks, stride)
    INVERTED_RESIDUAL_SETTING = [
        (1, 16, 1, 1),   # Stage 1 (features 1)
        (6, 24, 2, 2),   # Stage 2 (features 2-3)
        (6, 32, 3, 2),   # Stage 3 (features 4-6)
        (6, 64, 4, 2),   # Stage 4 (features 7-10)
        (6, 96, 3, 1),   # Stage 5 (features 11-13)
        (6, 160, 3, 2),  # Stage 6 (features 14-16)
        (6, 320, 1, 1),  # Stage 7 (features 17)
    ]

    def __init__(self):
        # Build features index -> (stage, layer) mapping
        self.features_to_stage: Dict[int, Tuple[int, int]] = {}
        features_idx = 1
        for stage_idx, (t, c, n, s) in enumerate(self.INVERTED_RESIDUAL_SETTING):
            for layer_idx in range(n):
                self.features_to_stage[features_idx] = (stage_idx + 1, layer_idx)
                features_idx += 1

    def map_key(self, pytorch_key: str) -> Optional[str]:
        """Map a PyTorch weight key to MLX model key."""

        # Stem: fx.features.0.0 -> backbone.stem.conv, fx.features.0.1 -> backbone.stem.bn
        if pytorch_key.startswith('fx.features.0.'):
            rest = pytorch_key[len('fx.features.0.'):]
            if rest.startswith('0.'):
                # Conv weight
                return 'backbone.stem.conv.' + rest[2:]
            elif rest.startswith('1.'):
                # BatchNorm
                return 'backbone.stem.bn.' + rest[2:]

        # Inverted residuals: fx.features.N.conv.X.Y
        match = re.match(r'fx\.features\.(\d+)\.conv\.(.+)', pytorch_key)
        if match:
            features_idx = int(match.group(1))
            rest = match.group(2)

            if features_idx not in self.features_to_stage:
                # Skip features.18 (final 1x1 conv, not used in RetinaFace)
                return None

            stage, layer = self.features_to_stage[features_idx]

            # Get expand_ratio for this stage
            t = self.INVERTED_RESIDUAL_SETTING[stage - 1][0]

            if t == 1:
                # No expansion block (stage 1, first layer)
                # Structure: conv.0.0 = dw conv, conv.0.1 = dw bn, conv.1 = pw conv, conv.2 = pw bn
                if rest.startswith('0.0.'):
                    return f'backbone.stage{stage}.layers.{layer}.depthwise.' + rest[4:]
                elif rest.startswith('0.1.'):
                    return f'backbone.stage{stage}.layers.{layer}.bn_dw.' + rest[4:]
                elif rest.startswith('1.'):
                    return f'backbone.stage{stage}.layers.{layer}.project.' + rest[2:]
                elif rest.startswith('2.'):
                    return f'backbone.stage{stage}.layers.{layer}.bn_proj.' + rest[2:]
            else:
                # With expansion block
                # Structure: conv.0.0 = expand conv, conv.0.1 = expand bn,
                #           conv.1.0 = dw conv, conv.1.1 = dw bn,
                #           conv.2 = project conv, conv.3 = project bn
                if rest.startswith('0.0.'):
                    return f'backbone.stage{stage}.layers.{layer}.expand.layers.0.' + rest[4:]
                elif rest.startswith('0.1.'):
                    return f'backbone.stage{stage}.layers.{layer}.expand.layers.1.' + rest[4:]
                elif rest.startswith('1.0.'):
                    return f'backbone.stage{stage}.layers.{layer}.depthwise.' + rest[4:]
                elif rest.startswith('1.1.'):
                    return f'backbone.stage{stage}.layers.{layer}.bn_dw.' + rest[4:]
                elif rest.startswith('2.'):
                    return f'backbone.stage{stage}.layers.{layer}.project.' + rest[2:]
                elif rest.startswith('3.'):
                    return f'backbone.stage{stage}.layers.{layer}.bn_proj.' + rest[2:]

        # Final 1x1 conv (features.18) - needed for RetinaFace MobileNetV2
        # fx.features.18.0.weight -> backbone.final_conv.conv.weight
        # fx.features.18.1.* -> backbone.final_conv.bn.*
        if pytorch_key.startswith('fx.features.18.'):
            rest = pytorch_key[len('fx.features.18.'):]
            if rest.startswith('0.'):
                return 'backbone.final_conv.conv.' + rest[2:]
            elif rest.startswith('1.'):
                return 'backbone.final_conv.bn.' + rest[2:]

        # FPN layers: fpn.output1.0.weight -> fpn.output1.conv.weight
        #             fpn.output1.1.weight -> fpn.output1.bn.weight
        match = re.match(r'fpn\.(output\d+|merge\d+)\.(\d+)\.(.+)', pytorch_key)
        if match:
            layer_name = match.group(1)
            sub_idx = int(match.group(2))
            param = match.group(3)
            if sub_idx == 0:
                return f'fpn.{layer_name}.conv.{param}'
            elif sub_idx == 1:
                return f'fpn.{layer_name}.bn.{param}'

        # SSH context modules: same structure, map sequential indices
        match = re.match(r'ssh(\d+)\.(\w+)\.(\d+)\.(.+)', pytorch_key)
        if match:
            ssh_idx = match.group(1)
            conv_name = match.group(2)
            sub_idx = int(match.group(3))
            param = match.group(4)
            if sub_idx == 0:
                return f'ssh{ssh_idx}.{conv_name}.conv.{param}'
            elif sub_idx == 1:
                return f'ssh{ssh_idx}.{conv_name}.bn.{param}'

        # Detection heads: class_head.class_head.0.weight -> class_head.class_head.layers.0.weight
        # MLX nn.Sequential stores layers in 'layers' attribute
        match = re.match(r'(class_head|bbox_head|landmark_head)\.\1\.(\d+)\.(.+)', pytorch_key)
        if match:
            head_type = match.group(1)
            idx = match.group(2)
            param = match.group(3)
            return f'{head_type}.{head_type}.layers.{idx}.{param}'

        return None


def convert_retinaface_mnetv2(
    input_path: str,
    output_path: str,
    verbose: bool = True,
) -> str:
    """
    Convert RetinaFace MobileNetV2 weights from PyTorch to MLX format.

    Args:
        input_path: Path to PyTorch .pth file
        output_path: Path for output .safetensors file
        verbose: Print conversion details

    Returns:
        SHA-256 hash of the output file
    """
    print(f'\nConverting: {input_path}')
    print(f'Output: {output_path}')

    # Load PyTorch weights
    state_dict = torch.load(input_path, map_location='cpu', weights_only=True)

    # Handle nested state_dict (some models wrap in 'state_dict' key)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    # Create key mapper
    mapper = RetinaFaceMNetV2KeyMapper()

    # Convert weights
    mlx_weights = {}
    mapped_count = 0
    skipped_count = 0
    transposed_count = 0

    for pytorch_key, tensor in state_dict.items():
        # Skip num_batches_tracked (not needed for inference)
        if 'num_batches_tracked' in pytorch_key:
            skipped_count += 1
            continue

        # Map key
        mlx_key = mapper.map_key(pytorch_key)

        if mlx_key is None:
            if verbose:
                print(f'  [SKIP] {pytorch_key}')
            skipped_count += 1
            continue

        # Convert tensor to numpy
        weight = tensor.numpy()

        # Transpose conv weights
        if 'weight' in mlx_key and weight.ndim == 4:
            original_shape = weight.shape
            weight = transpose_conv_weight(weight)
            transposed_count += 1
            if verbose:
                print(f'  [TRANSPOSE] {pytorch_key} -> {mlx_key}: {original_shape} -> {weight.shape}')
        elif verbose:
            print(f'  [MAP] {pytorch_key} -> {mlx_key}: {weight.shape}')

        mlx_weights[mlx_key] = weight
        mapped_count += 1

    print(f'\nConversion summary:')
    print(f'  Mapped: {mapped_count}')
    print(f'  Transposed: {transposed_count}')
    print(f'  Skipped: {skipped_count}')

    # Save as safetensors
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    save_file(mlx_weights, output_path)

    # Compute hash
    sha256 = compute_sha256(output_path)
    print(f'\nSHA-256: {sha256}')

    return sha256


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch weights to MLX format')
    parser.add_argument('--model', '-m', type=str, help='Model to convert')
    parser.add_argument('--input', '-i', type=str, help='Input PyTorch .pth file')
    parser.add_argument('--output', '-o', type=str, help='Output .safetensors file')
    parser.add_argument('--output-dir', type=str, default='weights_mlx_v2', help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.model == 'retinaface_mnet_v2':
        input_path = args.input or 'weights_pytorch/retinaface_mnet_v2.pth'
        output_path = args.output or os.path.join(args.output_dir, 'retinaface_mnet_v2.safetensors')
        convert_retinaface_mnetv2(input_path, output_path, args.verbose)
    else:
        parser.print_help()
        print('\nSupported models:')
        print('  retinaface_mnet_v2')


if __name__ == '__main__':
    main()
