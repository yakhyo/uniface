#!/usr/bin/env python3
# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo
#
# Weight Conversion Script: PyTorch -> MLX (safetensors)

"""
Convert PyTorch model weights to MLX-compatible safetensors format.

This script handles the necessary weight transformations:
1. Conv2d weights: PyTorch OIHW -> MLX OHWI (transpose axes 0,2,3,1)
2. Linear weights: No change needed
3. BatchNorm: No change needed

Usage:
    python scripts/convert_weights_to_mlx.py --input model.pth --output model.safetensors
    python scripts/convert_weights_to_mlx.py --all  # Convert all models
"""

import argparse
import hashlib
import os
from pathlib import Path
from typing import Dict

import numpy as np

# Check for required packages
try:
    import torch
except ImportError:
    print('PyTorch is required for weight conversion. Install with: pip install torch')
    exit(1)

try:
    from safetensors.numpy import save_file
except ImportError:
    print('safetensors is required. Install with: pip install safetensors')
    exit(1)


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def convert_conv_weight(weight: np.ndarray) -> np.ndarray:
    """
    Convert Conv2d weight from PyTorch format to MLX format.

    PyTorch Conv2d: (out_channels, in_channels, height, width) - OIHW
    MLX Conv2d: (out_channels, height, width, in_channels) - OHWI

    Args:
        weight: Conv2d weight in PyTorch format.

    Returns:
        Conv2d weight in MLX format.
    """
    if weight.ndim == 4:
        # Transpose from OIHW to OHWI
        return weight.transpose(0, 2, 3, 1)
    return weight


def should_transpose_conv(key: str, weight: np.ndarray) -> bool:
    """
    Determine if a weight tensor should be transposed as a Conv2d weight.

    Args:
        key: Parameter name.
        weight: Weight tensor.

    Returns:
        True if this is a Conv2d weight that needs transposition.
    """
    # Check if it's a 4D tensor (conv weight)
    if weight.ndim != 4:
        return False

    # Common patterns for conv weights
    conv_patterns = [
        'conv',
        'downsample.0',
        'features',
        'stem',
        'depthwise',
        'pointwise',
        'expand',
        'project',
        'lateral',
        'output',
        'branch',
        'cls',
        'bbox',
        'landmark',
    ]

    key_lower = key.lower()

    # Must be a weight, not bias
    if 'weight' not in key_lower:
        return False

    # Check for conv patterns
    for pattern in conv_patterns:
        if pattern in key_lower:
            return True

    # Default: transpose any 4D weight tensor
    return True


def convert_pytorch_to_mlx(
    state_dict: Dict[str, torch.Tensor],
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Convert PyTorch state dict to MLX-compatible format.

    Args:
        state_dict: PyTorch state dictionary.
        verbose: Print conversion details.

    Returns:
        Dictionary of numpy arrays ready for safetensors.
    """
    mlx_weights = {}
    transposed_count = 0
    total_count = 0

    for key, tensor in state_dict.items():
        # Convert to numpy
        np_array = tensor.cpu().numpy()
        total_count += 1

        # Check if we need to transpose Conv2d weights
        if should_transpose_conv(key, np_array):
            np_array = convert_conv_weight(np_array)
            transposed_count += 1
            if verbose:
                print(f'  [TRANSPOSE] {key}: {tensor.shape} -> {np_array.shape}')
        elif verbose:
            print(f'  [KEEP]      {key}: {np_array.shape}')

        mlx_weights[key] = np_array

    if verbose:
        print(f'\nConverted {total_count} tensors, transposed {transposed_count} conv weights')

    return mlx_weights


def convert_single_model(
    input_path: str,
    output_path: str,
    verbose: bool = True,
) -> str:
    """
    Convert a single PyTorch model to MLX safetensors format.

    Args:
        input_path: Path to PyTorch .pth file.
        output_path: Path for output .safetensors file.
        verbose: Print conversion details.

    Returns:
        SHA-256 hash of the output file.
    """
    print(f'\nConverting: {input_path}')
    print(f'Output: {output_path}')

    # Load PyTorch weights
    state_dict = torch.load(input_path, map_location='cpu', weights_only=True)

    # Handle nested state dicts (e.g., {'model': {...}, 'optimizer': {...}})
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model' in state_dict:
        state_dict = state_dict['model']

    # Convert to MLX format
    mlx_weights = convert_pytorch_to_mlx(state_dict, verbose=verbose)

    # Save as safetensors
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    save_file(mlx_weights, output_path)

    # Compute hash
    sha256 = compute_sha256(output_path)
    print(f'SHA-256: {sha256}')

    return sha256


def convert_all_models(
    input_dir: str = 'weights_pytorch',
    output_dir: str = 'weights_mlx',
    verbose: bool = False,
) -> Dict[str, str]:
    """
    Convert all PyTorch models in a directory to MLX format.

    Args:
        input_dir: Directory containing PyTorch .pth files.
        output_dir: Directory for output .safetensors files.
        verbose: Print detailed conversion info.

    Returns:
        Dictionary mapping model names to SHA-256 hashes.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    hashes = {}

    # Find all .pth files
    pth_files = list(input_path.glob('*.pth'))

    if not pth_files:
        print(f'No .pth files found in {input_dir}')
        return hashes

    print(f'Found {len(pth_files)} PyTorch models to convert')

    for pth_file in pth_files:
        model_name = pth_file.stem
        output_file = output_path / f'{model_name}.safetensors'

        try:
            sha256 = convert_single_model(
                str(pth_file),
                str(output_file),
                verbose=verbose,
            )
            hashes[model_name] = sha256
        except Exception as e:
            print(f'ERROR converting {pth_file}: {e}')
            continue

    # Print summary
    print('\n' + '=' * 60)
    print('Conversion Summary')
    print('=' * 60)
    print(f'Successfully converted: {len(hashes)}/{len(pth_files)} models')
    print('\nSHA-256 Hashes (for constants.py):')
    print('-' * 60)
    for model_name, sha256 in sorted(hashes.items()):
        print(f"    '{model_name}': '{sha256}',")

    return hashes


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch weights to MLX safetensors format')
    parser.add_argument('--input', '-i', type=str, help='Input PyTorch .pth file')
    parser.add_argument('--output', '-o', type=str, help='Output .safetensors file')
    parser.add_argument('--all', action='store_true', help='Convert all models in weights_pytorch/ directory')
    parser.add_argument('--input-dir', type=str, default='weights_pytorch', help='Input directory for --all mode')
    parser.add_argument('--output-dir', type=str, default='weights_mlx', help='Output directory for --all mode')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed conversion info')

    args = parser.parse_args()

    if args.all:
        convert_all_models(args.input_dir, args.output_dir, args.verbose)
    elif args.input and args.output:
        convert_single_model(args.input, args.output, args.verbose)
    else:
        parser.print_help()
        print('\nExamples:')
        print('  python convert_weights_to_mlx.py -i model.pth -o model.safetensors')
        print('  python convert_weights_to_mlx.py --all --input-dir weights_pytorch/')


if __name__ == '__main__':
    main()
