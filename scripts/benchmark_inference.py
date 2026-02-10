"""Benchmark inference speed for PatchCore models.

Compare inference speed between:
- Lightning checkpoint (.ckpt) - PyTorch
- ONNX Runtime
- TensorRT (if available)

Usage:
    # Benchmark specific model
    python scripts/benchmark_inference.py \
        --checkpoint output/Patchcore/GoodsAD/cigarette_box/v0/model.ckpt \
        --onnx models/onnx/GoodsAD/cigarette_box/model.onnx

    # Benchmark all models in directory
    python scripts/benchmark_inference.py \
        --checkpoint-dir output/Patchcore \
        --onnx-dir models/onnx

    # Adjust iterations for more accurate measurement
    python scripts/benchmark_inference.py \
        --checkpoint-dir output/Patchcore \
        --onnx-dir models/onnx \
        --warmup 10 \
        --iterations 100
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
PROJ_ROOT = SCRIPT_PATH.parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    throughput: float  # images/second
    memory_mb: Optional[float] = None

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.mean_ms:.2f} ± {self.std_ms:.2f} ms "
            f"(min: {self.min_ms:.2f}, max: {self.max_ms:.2f}) "
            f"[{self.throughput:.1f} img/s]"
        )


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def benchmark_function(
    func: Callable,
    input_data: np.ndarray,
    warmup: int = 5,
    iterations: int = 50,
    name: str = "Unknown",
) -> BenchmarkResult:
    """Benchmark a function with given input.

    Args:
        func: Function to benchmark (should accept input_data and return result)
        input_data: Input data to pass to function
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
        name: Name for this benchmark

    Returns:
        BenchmarkResult with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        _ = func(input_data)

    # Timed iterations
    gc.collect()
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        _ = func(input_data)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    times = np.array(times)
    memory_mb = get_memory_usage_mb()

    return BenchmarkResult(
        name=name,
        mean_ms=float(times.mean()),
        std_ms=float(times.std()),
        min_ms=float(times.min()),
        max_ms=float(times.max()),
        throughput=1000.0 / times.mean(),  # images per second
        memory_mb=memory_mb,
    )


def load_checkpoint_model(checkpoint_path: Path, device: str = "cpu"):
    """Load PatchCore model from Lightning checkpoint.

    Returns:
        Model object and predict function
    """
    try:
        import torch
        from anomalib.models import Patchcore
    except ImportError as e:
        raise ImportError(f"anomalib or torch not installed: {e}")

    print(f"  Loading checkpoint: {checkpoint_path}")
    model = Patchcore.load_from_checkpoint(str(checkpoint_path))
    model.eval()

    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()

    def predict_fn(image: np.ndarray):
        """Predict function for checkpoint model."""
        import cv2

        # Preprocess
        img = cv2.resize(image, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        with torch.no_grad():
            tensor = torch.from_numpy(img).float()
            if device == "cuda" and torch.cuda.is_available():
                tensor = tensor.cuda()
            output = model(tensor)

        return output

    return model, predict_fn


def load_onnx_model(onnx_path: Path, device: str = "cpu"):
    """Load PatchCore ONNX model.

    Returns:
        Session and predict function
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")

    print(f"  Loading ONNX: {onnx_path}")

    # Set providers
    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=providers,
    )

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    def predict_fn(image: np.ndarray):
        """Predict function for ONNX model."""
        import cv2

        # Preprocess
        img = cv2.resize(image, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0).astype(np.float32)

        outputs = session.run(output_names, {input_name: img})
        return outputs

    return session, predict_fn


def load_tensorrt_model(tensorrt_path: Path, device: str = "cuda"):
    """Load TensorRT engine.

    Returns:
        Engine and predict function
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        raise ImportError("TensorRT or pycuda not installed")

    print(f"  Loading TensorRT: {tensorrt_path}")

    # TensorRT loading is more complex, implement as needed
    raise NotImplementedError("TensorRT benchmark not yet implemented")


def find_model_pairs(
    checkpoint_dir: Path,
    onnx_dir: Path,
) -> List[Tuple[str, str, Optional[Path], Optional[Path]]]:
    """Find matching checkpoint and ONNX model pairs.

    Returns:
        List of (dataset, category, checkpoint_path, onnx_path)
    """
    pairs = []
    seen = set()

    # Find checkpoints
    patchcore_dir = checkpoint_dir / "Patchcore" if (checkpoint_dir / "Patchcore").exists() else checkpoint_dir

    if patchcore_dir.exists():
        for dataset_dir in patchcore_dir.iterdir():
            if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
                continue
            if dataset_dir.name in ["eval", "predictions"]:
                continue

            for category_dir in dataset_dir.iterdir():
                if not category_dir.is_dir():
                    continue

                # Find latest version
                versions = []
                for v_dir in category_dir.iterdir():
                    if v_dir.is_dir() and v_dir.name.startswith("v"):
                        try:
                            versions.append((int(v_dir.name[1:]), v_dir))
                        except ValueError:
                            continue

                if versions:
                    latest = max(versions, key=lambda x: x[0])[1]
                    ckpt = latest / "model.ckpt"
                    if ckpt.exists():
                        key = (dataset_dir.name, category_dir.name)
                        if key not in seen:
                            seen.add(key)
                            onnx_path = onnx_dir / dataset_dir.name / category_dir.name / "model.onnx"
                            pairs.append((
                                dataset_dir.name,
                                category_dir.name,
                                ckpt,
                                onnx_path if onnx_path.exists() else None,
                            ))

    # Find ONNX models without checkpoints
    if onnx_dir.exists():
        for dataset_dir in onnx_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            for category_dir in dataset_dir.iterdir():
                if not category_dir.is_dir():
                    continue
                onnx_path = category_dir / "model.onnx"
                if onnx_path.exists():
                    key = (dataset_dir.name, category_dir.name)
                    if key not in seen:
                        seen.add(key)
                        pairs.append((dataset_dir.name, category_dir.name, None, onnx_path))

    return sorted(pairs, key=lambda x: (x[0], x[1]))


def create_dummy_image(size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Create a dummy BGR image for benchmarking."""
    return np.random.randint(0, 255, (*size, 3), dtype=np.uint8)


def print_results_table(results: Dict[str, List[BenchmarkResult]]):
    """Print results in a formatted table."""
    print()
    print("=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)

    # Header
    print(f"{'Model':<40} {'Format':<12} {'Mean (ms)':<12} {'Std (ms)':<10} "
          f"{'Throughput':<12} {'Speedup':<10}")
    print("-" * 100)

    for model_key, model_results in results.items():
        base_time = None

        for i, result in enumerate(model_results):
            if i == 0:
                base_time = result.mean_ms

            speedup = base_time / result.mean_ms if base_time else 1.0
            speedup_str = f"{speedup:.2f}x" if speedup != 1.0 else "baseline"

            if i == 0:
                print(f"{model_key:<40} {result.name:<12} {result.mean_ms:<12.2f} "
                      f"{result.std_ms:<10.2f} {result.throughput:<12.1f} {speedup_str:<10}")
            else:
                print(f"{'':<40} {result.name:<12} {result.mean_ms:<12.2f} "
                      f"{result.std_ms:<10.2f} {result.throughput:<12.1f} {speedup_str:<10}")

        print()

    print("=" * 100)


def print_summary(results: Dict[str, List[BenchmarkResult]]):
    """Print summary statistics."""
    all_ckpt = []
    all_onnx = []

    for model_results in results.values():
        for result in model_results:
            if "ckpt" in result.name.lower() or "checkpoint" in result.name.lower():
                all_ckpt.append(result.mean_ms)
            elif "onnx" in result.name.lower():
                all_onnx.append(result.mean_ms)

    print("\nSUMMARY")
    print("-" * 50)

    if all_ckpt:
        print(f"Checkpoint average: {np.mean(all_ckpt):.2f} ms ({1000/np.mean(all_ckpt):.1f} img/s)")
    if all_onnx:
        print(f"ONNX average: {np.mean(all_onnx):.2f} ms ({1000/np.mean(all_onnx):.1f} img/s)")

    if all_ckpt and all_onnx:
        speedup = np.mean(all_ckpt) / np.mean(all_onnx)
        print(f"Average speedup: {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark PatchCore inference speed")

    # Single model options
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to single checkpoint file",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default=None,
        help="Path to single ONNX file",
    )

    # Directory options
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="output",
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--onnx-dir",
        type=str,
        default="models/onnx",
        help="Directory containing ONNX models",
    )

    # Benchmark options
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for inference",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Input image size for benchmark",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=None,
        help="Maximum number of models to benchmark",
    )

    args = parser.parse_args()

    # Prepare model pairs
    if args.checkpoint or args.onnx:
        # Single model mode
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
        onnx_path = Path(args.onnx) if args.onnx else None

        if checkpoint_path and not checkpoint_path.exists():
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        if onnx_path and not onnx_path.exists():
            print(f"Error: ONNX file not found: {onnx_path}")
            sys.exit(1)

        model_pairs = [("single", "model", checkpoint_path, onnx_path)]
    else:
        # Directory mode
        checkpoint_dir = Path(args.checkpoint_dir)
        onnx_dir = Path(args.onnx_dir)

        model_pairs = find_model_pairs(checkpoint_dir, onnx_dir)

        if not model_pairs:
            print("No models found. Check --checkpoint-dir and --onnx-dir paths.")
            sys.exit(1)

    if args.max_models:
        model_pairs = model_pairs[:args.max_models]

    print(f"Found {len(model_pairs)} model(s) to benchmark")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print(f"Device: {args.device}")
    print(f"Image size: {args.image_size[0]}x{args.image_size[1]}")
    print()

    # Create dummy input
    dummy_image = create_dummy_image(tuple(args.image_size))

    # Run benchmarks
    all_results: Dict[str, List[BenchmarkResult]] = {}

    for dataset, category, ckpt_path, onnx_path in model_pairs:
        model_key = f"{dataset}/{category}"
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {model_key}")
        print("=" * 60)

        model_results = []

        # Benchmark checkpoint
        if ckpt_path:
            try:
                model, predict_fn = load_checkpoint_model(ckpt_path, args.device)
                result = benchmark_function(
                    predict_fn,
                    dummy_image,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    name="Checkpoint",
                )
                model_results.append(result)
                print(f"  {result}")

                # Clean up
                del model, predict_fn
                gc.collect()

                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

            except Exception as e:
                print(f"  Checkpoint failed: {e}")

        # Benchmark ONNX
        if onnx_path:
            try:
                session, predict_fn = load_onnx_model(onnx_path, args.device)
                result = benchmark_function(
                    predict_fn,
                    dummy_image,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    name="ONNX",
                )
                model_results.append(result)
                print(f"  {result}")

                # Clean up
                del session, predict_fn
                gc.collect()

            except Exception as e:
                print(f"  ONNX failed: {e}")

        if model_results:
            all_results[model_key] = model_results

    # Print results table
    if all_results:
        print_results_table(all_results)
        print_summary(all_results)
    else:
        print("\nNo successful benchmarks to report.")


if __name__ == "__main__":
    main()
