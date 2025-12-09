#!/usr/bin/env python3
# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo
#
# Download PyTorch weights for MLX conversion

"""
Download PyTorch model weights from original repositories.

This script downloads the original PyTorch .pth weights needed for
converting to MLX safetensors format.

Usage:
    python scripts/download_pytorch_weights.py --all
    python scripts/download_pytorch_weights.py --model retinaface_mnet_v2
"""

import argparse
import hashlib
import os
from typing import Dict, Optional

import requests
from tqdm import tqdm

# Output directory for downloaded weights
OUTPUT_DIR = 'weights_pytorch'

# PyTorch weight sources
# Format: model_name -> (url, expected_sha256 or None)
PYTORCH_WEIGHT_URLS: Dict[str, tuple] = {
    # ==========================================================================
    # RetinaFace - from yakhyo/retinaface-pytorch
    # https://github.com/yakhyo/retinaface-pytorch/releases
    # ==========================================================================
    'retinaface_mnet025': (
        'https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv1_0.25.pth',
        None,
    ),
    'retinaface_mnet050': (
        'https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv1_0.50.pth',
        None,
    ),
    'retinaface_mnet_v1': (
        'https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv1.pth',
        None,
    ),
    'retinaface_mnet_v2': (
        'https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv2.pth',
        None,
    ),
    # Note: ResNet models not available in yakhyo's repo, need alternate source
    # 'retinaface_r18': not available as PyTorch
    # 'retinaface_r34': not available as PyTorch
    # ==========================================================================
    # MobileFace / SphereFace - from yakhyo/face-recognition
    # https://github.com/yakhyo/face-recognition/releases
    # Note: These have _mcp suffix (margin-based classification pretraining)
    # ==========================================================================
    'mobilenetv1_025': (
        'https://github.com/yakhyo/face-recognition/releases/download/v0.0.1/mobilenetv1_0.25_mcp.pth',
        None,
    ),
    'mobilenetv2': (
        'https://github.com/yakhyo/face-recognition/releases/download/v0.0.1/mobilenetv2_mcp.pth',
        None,
    ),
    'mobilenetv3_small': (
        'https://github.com/yakhyo/face-recognition/releases/download/v0.0.1/mobilenetv3_small_mcp.pth',
        None,
    ),
    'mobilenetv3_large': (
        'https://github.com/yakhyo/face-recognition/releases/download/v0.0.1/mobilenetv3_large_mcp.pth',
        None,
    ),
    'sphere20': (
        'https://github.com/yakhyo/face-recognition/releases/download/v0.0.1/sphere20_mcp.pth',
        None,
    ),
    # Note: sphere36 not available in yakhyo's repo
    # ==========================================================================
    # Emotion (DDAMFN) - from SainingZhang/DDAMFN
    # These are TorchScript .script files, already available in uniface releases
    # ==========================================================================
    'affecnet7': (
        'https://github.com/yakhyo/uniface/releases/download/weights/affecnet7.script',
        None,
    ),
    'affecnet8': (
        'https://github.com/yakhyo/uniface/releases/download/weights/affecnet8.script',
        None,
    ),
}

# Models that are NOT available as PyTorch weights (only ONNX)
# These would need conversion from ONNX or finding alternate sources:
UNAVAILABLE_MODELS = [
    'retinaface_r18',  # ResNet18 backbone - ONNX only
    'retinaface_r34',  # ResNet34 backbone - ONNX only
    'sphere36',  # SphereFace 36 - not in releases
    'arcface_mnet',  # ArcFace MobileNet - InsightFace ONNX only
    'arcface_resnet',  # ArcFace ResNet - InsightFace ONNX only
    'scrfd_10g',  # SCRFD 10G - InsightFace ONNX only
    'scrfd_500m',  # SCRFD 500M - InsightFace ONNX only
    'yolov5s_face',  # YOLOv5s Face - ONNX only in releases
    'yolov5m_face',  # YOLOv5m Face - ONNX only in releases
    'age_gender',  # AgeGender - InsightFace ONNX only
    '2d106det',  # Landmark 106 - InsightFace ONNX only
]


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def download_file(url: str, output_path: str, expected_sha256: Optional[str] = None) -> bool:
    """
    Download a file with progress bar.

    Args:
        url: URL to download from.
        output_path: Local path to save the file.
        expected_sha256: Optional SHA-256 hash to verify.

    Returns:
        True if download successful, False otherwise.
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(output_path)) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Verify hash if provided
        if expected_sha256:
            actual_sha256 = compute_sha256(output_path)
            if actual_sha256 != expected_sha256:
                print('  WARNING: SHA-256 mismatch!')
                print(f'    Expected: {expected_sha256}')
                print(f'    Got:      {actual_sha256}')
                return False

        return True

    except requests.exceptions.RequestException as e:
        print(f'  ERROR: Failed to download {url}')
        print(f'    {e}')
        return False


def download_model(model_name: str, output_dir: str = OUTPUT_DIR) -> bool:
    """
    Download a single model's PyTorch weights.

    Args:
        model_name: Name of the model to download.
        output_dir: Directory to save weights.

    Returns:
        True if successful, False otherwise.
    """
    if model_name not in PYTORCH_WEIGHT_URLS:
        print(f'Unknown model: {model_name}')
        print(f'Available models: {list(PYTORCH_WEIGHT_URLS.keys())}')
        return False

    url, expected_sha256 = PYTORCH_WEIGHT_URLS[model_name]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Determine output filename
    ext = os.path.splitext(url)[1]
    if ext in ['.pt', '.pth', '.script']:
        output_path = os.path.join(output_dir, f'{model_name}.pth')
    else:
        output_path = os.path.join(output_dir, f'{model_name}{ext}')

    print(f'\nDownloading: {model_name}')
    print(f'  URL: {url}')
    print(f'  Output: {output_path}')

    if os.path.exists(output_path):
        print('  File already exists, skipping.')
        return True

    return download_file(url, output_path, expected_sha256)


def download_all(output_dir: str = OUTPUT_DIR) -> Dict[str, bool]:
    """
    Download all available PyTorch weights.

    Args:
        output_dir: Directory to save weights.

    Returns:
        Dictionary mapping model names to success status.
    """
    results = {}

    print('=' * 60)
    print('Downloading PyTorch Weights for MLX Conversion')
    print('=' * 60)
    print(f'Output directory: {output_dir}')
    print(f'Models to download: {len(PYTORCH_WEIGHT_URLS)}')

    for model_name in PYTORCH_WEIGHT_URLS:
        success = download_model(model_name, output_dir)
        results[model_name] = success

    # Print summary
    print('\n' + '=' * 60)
    print('Download Summary')
    print('=' * 60)

    success_count = sum(1 for v in results.values() if v)
    fail_count = len(results) - success_count

    print(f'Successful: {success_count}/{len(results)}')
    if fail_count > 0:
        print(f'Failed: {fail_count}')
        print('\nFailed models:')
        for model, success in results.items():
            if not success:
                print(f'  - {model}')

    print('\n' + '=' * 60)
    print('Next Steps')
    print('=' * 60)
    print('Run the conversion script:')
    print(f'  python scripts/convert_weights_to_mlx.py --all --input-dir {output_dir}')

    return results


def list_models():
    """Print available models."""
    print('=' * 70)
    print('Available PyTorch Models for Download')
    print('=' * 70)

    categories = {
        'RetinaFace (Detection)': ['retinaface_'],
        'MobileFace (Recognition)': ['mobilenet'],
        'SphereFace (Recognition)': ['sphere'],
        'Emotion (DDAMFN)': ['affecnet'],
    }

    for category, prefixes in categories.items():
        models = [m for m in PYTORCH_WEIGHT_URLS if any(m.startswith(p) for p in prefixes)]
        if models:
            print(f'\n{category}:')
            for model in models:
                print(f'  ✓ {model}')

    print('\n' + '=' * 70)
    print('Models NOT Available as PyTorch (ONNX only)')
    print('=' * 70)
    print('These models are only available in ONNX format from InsightFace:')
    for model in UNAVAILABLE_MODELS:
        print(f'  ✗ {model}')

    print('\n' + '-' * 70)
    print(f'Total available: {len(PYTORCH_WEIGHT_URLS)} models')
    print(f'Total unavailable: {len(UNAVAILABLE_MODELS)} models')
    print('-' * 70)


def main():
    parser = argparse.ArgumentParser(description='Download PyTorch weights for MLX conversion')
    parser.add_argument('--model', '-m', type=str, help='Specific model to download')
    parser.add_argument('--all', action='store_true', help='Download all models')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--output-dir', '-o', type=str, default=OUTPUT_DIR, help='Output directory')

    args = parser.parse_args()

    if args.list:
        list_models()
    elif args.all:
        download_all(args.output_dir)
    elif args.model:
        download_model(args.model, args.output_dir)
    else:
        parser.print_help()
        print('\nExamples:')
        print('  python download_pytorch_weights.py --list')
        print('  python download_pytorch_weights.py --model retinaface_mnet_v2')
        print('  python download_pytorch_weights.py --all')


if __name__ == '__main__':
    main()
