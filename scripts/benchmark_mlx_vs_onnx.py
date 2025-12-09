#!/usr/bin/env python3
# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo
#
# MLX vs ONNX Performance Benchmark Script

"""
Benchmark MLX vs ONNX Runtime performance on Apple Silicon.

Measures:
- Inference latency (ms)
- Throughput (FPS)
- Memory usage (MB)

Usage:
    python scripts/benchmark_mlx_vs_onnx.py --model retinaface --iterations 100
    python scripts/benchmark_mlx_vs_onnx.py --all
"""

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    model_name: str
    backend: str
    warmup_iterations: int
    benchmark_iterations: int
    avg_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_fps: float
    memory_mb: Optional[float] = None

    def __str__(self) -> str:
        lines = [
            f'{self.model_name} ({self.backend})',
            f'  Latency: {self.avg_latency_ms:.2f} ± {self.std_latency_ms:.2f} ms',
            f'  Min/Max: {self.min_latency_ms:.2f} / {self.max_latency_ms:.2f} ms',
            f'  Throughput: {self.throughput_fps:.1f} FPS',
        ]
        if self.memory_mb:
            lines.append(f'  Memory: {self.memory_mb:.1f} MB')
        return '\n'.join(lines)


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024 / 1024  # Convert to MB (macOS uses bytes)
    except Exception:
        return 0.0


def run_benchmark(
    inference_fn: Callable,
    input_data: np.ndarray,
    warmup: int = 10,
    iterations: int = 100,
) -> BenchmarkResult:
    """Run benchmark on an inference function."""
    # Warmup
    for _ in range(warmup):
        _ = inference_fn(input_data)

    # Force garbage collection
    gc.collect()

    # Measure memory before
    mem_before = get_memory_usage_mb()

    # Benchmark
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = inference_fn(input_data)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    # Measure memory after
    mem_after = get_memory_usage_mb()

    latencies = np.array(latencies)

    return BenchmarkResult(
        model_name='',  # To be filled by caller
        backend='',  # To be filled by caller
        warmup_iterations=warmup,
        benchmark_iterations=iterations,
        avg_latency_ms=float(np.mean(latencies)),
        std_latency_ms=float(np.std(latencies)),
        min_latency_ms=float(np.min(latencies)),
        max_latency_ms=float(np.max(latencies)),
        throughput_fps=1000.0 / float(np.mean(latencies)),
        memory_mb=mem_after - mem_before if mem_after > mem_before else None,
    )


def create_test_image(height: int = 640, width: int = 640) -> np.ndarray:
    """Create a random test image."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def benchmark_detection(
    model_type: str = 'retinaface',
    warmup: int = 10,
    iterations: int = 100,
) -> Dict[str, BenchmarkResult]:
    """Benchmark detection models."""
    results = {}
    image = create_test_image(640, 640)

    print(f'\nBenchmarking {model_type.upper()} Detection')
    print('-' * 40)

    # ONNX benchmark
    try:
        os.environ['UNIFACE_BACKEND'] = 'onnx'

        if model_type == 'retinaface':
            from uniface.detection.retinaface import RetinaFace

            detector = RetinaFace()
        elif model_type == 'scrfd':
            from uniface.detection.scrfd import SCRFD

            detector = SCRFD()
        elif model_type == 'yolov5':
            from uniface.detection.yolov5 import YOLOv5Face

            detector = YOLOv5Face()
        else:
            raise ValueError(f'Unknown model type: {model_type}')

        print(f'  Running ONNX benchmark ({iterations} iterations)...')
        result = run_benchmark(detector.detect, image, warmup, iterations)
        result.model_name = model_type
        result.backend = 'ONNX'
        results['onnx'] = result
        print(f'  ONNX: {result.avg_latency_ms:.2f} ms ({result.throughput_fps:.1f} FPS)')

    except Exception as e:
        print(f'  ONNX benchmark failed: {e}')

    # MLX benchmark
    try:
        os.environ['UNIFACE_BACKEND'] = 'mlx'

        if model_type == 'retinaface':
            from uniface.detection.retinaface_mlx import RetinaFaceMLX

            detector = RetinaFaceMLX()
        elif model_type == 'scrfd':
            from uniface.detection.scrfd_mlx import SCRFDMLX

            detector = SCRFDMLX()
        elif model_type == 'yolov5':
            from uniface.detection.yolov5_mlx import YOLOv5FaceMLX

            detector = YOLOv5FaceMLX()

        print(f'  Running MLX benchmark ({iterations} iterations)...')
        result = run_benchmark(detector.detect, image, warmup, iterations)
        result.model_name = model_type
        result.backend = 'MLX'
        results['mlx'] = result
        print(f'  MLX: {result.avg_latency_ms:.2f} ms ({result.throughput_fps:.1f} FPS)')

    except Exception as e:
        print(f'  MLX benchmark failed: {e}')

    return results


def benchmark_recognition(
    warmup: int = 10,
    iterations: int = 100,
) -> Dict[str, BenchmarkResult]:
    """Benchmark recognition models."""
    results = {}

    # Create test face image (112x112)
    face_image = create_test_image(112, 112)
    landmarks = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float32,
    )

    print('\nBenchmarking ArcFace Recognition')
    print('-' * 40)

    # ONNX benchmark
    try:
        os.environ['UNIFACE_BACKEND'] = 'onnx'
        from uniface.recognition.models import ArcFace

        recognizer = ArcFace()

        print(f'  Running ONNX benchmark ({iterations} iterations)...')

        def inference_fn(img):
            return recognizer.get_embedding(img, landmarks)

        result = run_benchmark(inference_fn, face_image, warmup, iterations)
        result.model_name = 'ArcFace'
        result.backend = 'ONNX'
        results['onnx'] = result
        print(f'  ONNX: {result.avg_latency_ms:.2f} ms ({result.throughput_fps:.1f} FPS)')

    except Exception as e:
        print(f'  ONNX benchmark failed: {e}')

    # MLX benchmark
    try:
        os.environ['UNIFACE_BACKEND'] = 'mlx'
        from uniface.recognition.models_mlx import ArcFaceMLX

        recognizer = ArcFaceMLX()

        print(f'  Running MLX benchmark ({iterations} iterations)...')

        def inference_fn(img):
            return recognizer.get_embedding(img, landmarks)

        result = run_benchmark(inference_fn, face_image, warmup, iterations)
        result.model_name = 'ArcFace'
        result.backend = 'MLX'
        results['mlx'] = result
        print(f'  MLX: {result.avg_latency_ms:.2f} ms ({result.throughput_fps:.1f} FPS)')

    except Exception as e:
        print(f'  MLX benchmark failed: {e}')

    return results


def print_comparison(results: Dict[str, Dict[str, BenchmarkResult]]) -> None:
    """Print benchmark comparison table."""
    print('\n' + '=' * 70)
    print('Performance Comparison Summary')
    print('=' * 70)
    print(f'{"Model":<20} {"ONNX (ms)":<15} {"MLX (ms)":<15} {"Speedup":<15}')
    print('-' * 70)

    for model_name, model_results in results.items():
        onnx_result = model_results.get('onnx')
        mlx_result = model_results.get('mlx')

        onnx_str = f'{onnx_result.avg_latency_ms:.2f}' if onnx_result else 'N/A'
        mlx_str = f'{mlx_result.avg_latency_ms:.2f}' if mlx_result else 'N/A'

        if onnx_result and mlx_result:
            speedup = onnx_result.avg_latency_ms / mlx_result.avg_latency_ms
            speedup_str = f'{speedup:.2f}x'
        else:
            speedup_str = 'N/A'

        print(f'{model_name:<20} {onnx_str:<15} {mlx_str:<15} {speedup_str:<15}')

    print('=' * 70)


def main():
    parser = argparse.ArgumentParser(description='Benchmark MLX vs ONNX performance')
    parser.add_argument(
        '--model',
        '-m',
        type=str,
        choices=['retinaface', 'scrfd', 'yolov5', 'arcface'],
        help='Model to benchmark',
    )
    parser.add_argument('--all', action='store_true', help='Benchmark all models')
    parser.add_argument('--warmup', '-w', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--iterations', '-n', type=int, default=100, help='Benchmark iterations')

    args = parser.parse_args()

    print('=' * 70)
    print('UniFace MLX vs ONNX Performance Benchmark')
    print('=' * 70)

    # Check platform
    import platform

    print(f'Platform: {platform.system()} {platform.machine()}')
    print(f'Python: {platform.python_version()}')

    try:
        import mlx.core as mx

        print(f'MLX: Available (default device: {mx.default_device()})')
    except ImportError:
        print('MLX: Not available')

    all_results = {}

    if args.all:
        # Benchmark all detection models
        for model in ['retinaface', 'scrfd', 'yolov5']:
            results = benchmark_detection(model, args.warmup, args.iterations)
            all_results[model] = results

        # Benchmark recognition
        results = benchmark_recognition(args.warmup, args.iterations)
        all_results['arcface'] = results

    elif args.model:
        if args.model in ['retinaface', 'scrfd', 'yolov5']:
            results = benchmark_detection(args.model, args.warmup, args.iterations)
            all_results[args.model] = results
        elif args.model == 'arcface':
            results = benchmark_recognition(args.warmup, args.iterations)
            all_results['arcface'] = results
    else:
        parser.print_help()
        print('\nExamples:')
        print('  python benchmark_mlx_vs_onnx.py --model retinaface')
        print('  python benchmark_mlx_vs_onnx.py --all --iterations 50')
        return

    # Print comparison
    if all_results:
        print_comparison(all_results)

    print('\nNote: MLX benchmarks require converted weights for accurate timing.')
    print('Without weights, the benchmark measures model initialization overhead.')


if __name__ == '__main__':
    main()
