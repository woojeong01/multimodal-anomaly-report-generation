"""Benchmark inference speed: Checkpoint vs ONNX.

Usage:
    python scripts/benchmark_inference.py \
        --checkpoint-dir /path/to/checkpoints \
        --onnx-dir models/onnx \
        --config configs/anomaly.yaml \
        --device cuda
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
PROJ_ROOT = SCRIPT_PATH.parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))


@dataclass
class BenchmarkResult:
    """Benchmark result."""
    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    throughput: float  # images/second
    total_time_s: float = 0.0

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.mean_ms:.2f} ± {self.std_ms:.2f} ms "
            f"[{self.throughput:.1f} img/s]"
        )


def benchmark_function(
    func,
    input_data: np.ndarray,
    warmup: int = 5,
    iterations: int = 50,
    name: str = "Unknown",
) -> BenchmarkResult:
    """Benchmark a function."""
    total_start = time.perf_counter()

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
        times.append((end - start) * 1000)

    total_time = time.perf_counter() - total_start
    times = np.array(times)

    return BenchmarkResult(
        name=name,
        mean_ms=float(times.mean()),
        std_ms=float(times.std()),
        min_ms=float(times.min()),
        max_ms=float(times.max()),
        throughput=1000.0 / times.mean(),
        total_time_s=total_time,
    )


def load_checkpoint_model(checkpoint_path: Path, device: str, input_size: Tuple[int, int]):
    """Load PatchCore from checkpoint."""
    import torch
    from anomalib.models import Patchcore

    print(f"  Loading checkpoint: {checkpoint_path}")
    model = Patchcore.load_from_checkpoint(str(checkpoint_path), map_location="cpu")
    model.eval()

    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()

    def predict_fn(image: np.ndarray):
        import cv2
        img = cv2.resize(image, (input_size[1], input_size[0]))
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


def load_onnx_model(model_dir: Path, device: str):
    """Load PatchCore ONNX (backbone + memory bank)."""
    from src.anomaly import PatchCoreOnnx

    print(f"  Loading ONNX: {model_dir}")

    model = PatchCoreOnnx(
        model_path=model_dir,
        device=device,
    )
    model.load_model()

    input_size = model.input_size
    print(f"  Input size: {input_size}")

    def predict_fn(image: np.ndarray):
        return model.predict(image)

    return model, predict_fn, input_size


def find_model_pairs(
    checkpoint_dir: Path,
    onnx_dir: Path,
    version: int | None = None,
    datasets: List[str] | None = None,
    categories: List[str] | None = None,
) -> List[Tuple[str, str, Optional[Path], Optional[Path]]]:
    """Find matching checkpoint and ONNX model pairs."""
    pairs = []
    seen = set()

    patchcore_dir = checkpoint_dir / "Patchcore" if (checkpoint_dir / "Patchcore").exists() else checkpoint_dir

    if patchcore_dir.exists():
        for dataset_dir in sorted(patchcore_dir.iterdir()):
            if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
                continue
            if dataset_dir.name in ["eval", "predictions"]:
                continue
            if datasets and dataset_dir.name not in datasets:
                continue

            for category_dir in sorted(dataset_dir.iterdir()):
                if not category_dir.is_dir():
                    continue
                if categories and category_dir.name not in categories:
                    continue

                ckpt = None
                if version is not None:
                    # Strict version: only use specified version, no fallback
                    candidate = category_dir / f"v{version}" / "model.ckpt"
                    if candidate.exists():
                        ckpt = candidate
                    else:
                        print(f"  Warning: v{version} not found for {dataset_dir.name}/{category_dir.name}, skipping")
                else:
                    # No version specified: use latest
                    versions = []
                    for v_dir in category_dir.iterdir():
                        if v_dir.is_dir() and v_dir.name.startswith("v"):
                            try:
                                versions.append((int(v_dir.name[1:]), v_dir))
                            except ValueError:
                                continue
                    if versions:
                        latest = max(versions, key=lambda x: x[0])[1]
                        candidate = latest / "model.ckpt"
                        if candidate.exists():
                            ckpt = candidate

                if ckpt:
                    key = (dataset_dir.name, category_dir.name)
                    if key not in seen:
                        seen.add(key)
                        onnx_path = onnx_dir / dataset_dir.name / category_dir.name
                        pairs.append((
                            dataset_dir.name,
                            category_dir.name,
                            ckpt,
                            onnx_path if (onnx_path / "backbone.onnx").exists() else None,
                        ))

    return sorted(pairs)


def create_dummy_image(size: Tuple[int, int] = (700, 700)) -> np.ndarray:
    """Create dummy BGR image."""
    return np.random.randint(0, 255, (*size, 3), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference speed")
    parser.add_argument("--checkpoint-dir", type=str, default="output")
    parser.add_argument("--onnx-dir", type=str, default="models/onnx")
    parser.add_argument("--config", type=str, default="configs/anomaly.yaml")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-models", type=int, default=None)
    args = parser.parse_args()

    # Load config
    config_version = None
    config_datasets = None
    config_categories = None
    input_size = (700, 700)

    if args.config:
        from src.utils.loaders import load_config
        config = load_config(args.config)
        config_datasets = config.get("data", {}).get("datasets")
        config_categories = config.get("data", {}).get("categories")
        config_version = config.get("predict", {}).get("version")
        config_input_size = config.get("data", {}).get("image_size")
        if config_input_size:
            input_size = tuple(config_input_size)

        print(f"Config: {args.config}")
        print(f"  Input size: {input_size}")
        print(f"  Version: v{config_version}" if config_version else "  Version: latest")
        print()

    # Find models
    checkpoint_dir = Path(args.checkpoint_dir)
    onnx_dir = Path(args.onnx_dir)

    model_pairs = find_model_pairs(
        checkpoint_dir, onnx_dir,
        version=config_version,
        datasets=config_datasets,
        categories=config_categories,
    )

    if args.max_models:
        model_pairs = model_pairs[:args.max_models]

    if not model_pairs:
        print("No models found!")
        return

    print(f"Found {len(model_pairs)} model(s)")
    print(f"Device: {args.device}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()

    # Create dummy image
    dummy_image = create_dummy_image(input_size)

    # Run benchmarks
    all_results: Dict[str, List[BenchmarkResult]] = {}

    for dataset, category, ckpt_path, onnx_path in model_pairs:
        model_key = f"{dataset}/{category}"
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {model_key}")
        if ckpt_path:
            # Extract version from path (e.g., .../v1/model.ckpt -> v1)
            version_name = ckpt_path.parent.name
            print(f"  Checkpoint version: {version_name}")
        print("=" * 60)

        model_results = []

        # Benchmark checkpoint
        if ckpt_path:
            try:
                model, predict_fn = load_checkpoint_model(ckpt_path, args.device, input_size)
                result = benchmark_function(
                    predict_fn, dummy_image,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    name="Checkpoint",
                )
                model_results.append(result)
                print(f"  {result}")

                del model, predict_fn
                gc.collect()
                if args.device == "cuda":
                    import torch
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"  Checkpoint failed: {e}")

        # Benchmark ONNX
        if onnx_path:
            try:
                model, predict_fn, _ = load_onnx_model(onnx_path, args.device)
                result = benchmark_function(
                    predict_fn, dummy_image,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    name="ONNX",
                )
                model_results.append(result)
                print(f"  {result}")

                del model, predict_fn
                gc.collect()

            except Exception as e:
                print(f"  ONNX failed: {e}")

        if model_results:
            all_results[model_key] = model_results

    # Print summary
    if all_results:
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(f"{'Model':<30} {'Checkpoint':>15} {'ONNX':>15} {'Speedup':>10}")
        print("-" * 80)

        all_ckpt_times = []
        all_onnx_times = []

        for model_key, results in all_results.items():
            ckpt_time = None
            onnx_time = None

            for r in results:
                if r.name == "Checkpoint":
                    ckpt_time = r.mean_ms
                    all_ckpt_times.append(ckpt_time)
                elif r.name == "ONNX":
                    onnx_time = r.mean_ms
                    all_onnx_times.append(onnx_time)

            ckpt_str = f"{ckpt_time:.2f} ms" if ckpt_time else "N/A"
            onnx_str = f"{onnx_time:.2f} ms" if onnx_time else "N/A"

            if ckpt_time and onnx_time:
                speedup = ckpt_time / onnx_time
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "N/A"

            print(f"{model_key:<30} {ckpt_str:>15} {onnx_str:>15} {speedup_str:>10}")

        print("-" * 80)

        if all_ckpt_times:
            avg_ckpt = np.mean(all_ckpt_times)
            print(f"{'Average Checkpoint':<30} {avg_ckpt:>14.2f} ms")
        if all_onnx_times:
            avg_onnx = np.mean(all_onnx_times)
            print(f"{'Average ONNX':<30} {' ':>15} {avg_onnx:>14.2f} ms")
        if all_ckpt_times and all_onnx_times:
            avg_speedup = np.mean(all_ckpt_times) / np.mean(all_onnx_times)
            print(f"{'Average Speedup':<30} {' ':>30} {avg_speedup:>9.2f}x")


if __name__ == "__main__":
    main()
