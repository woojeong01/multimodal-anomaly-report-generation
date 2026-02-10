"""Export PatchCore checkpoints to ONNX format.

Usage:
    # Export categories from config (recommended)
    python scripts/export_patchcore.py --checkpoint-dir output --output-dir models/onnx --config configs/anomaly.yaml

    # Export specific category
    python scripts/export_patchcore.py --checkpoint-dir output --output-dir models/onnx \
        --dataset GoodsAD --category cigarette_box

    # List available checkpoints
    python scripts/export_patchcore.py --checkpoint-dir output --list-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import torch

SCRIPT_PATH = Path(__file__).resolve()
PROJ_ROOT = SCRIPT_PATH.parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))


def find_patchcore_checkpoints(
    checkpoint_dir: Path,
    version: int | None = None,
    verbose: bool = False,
) -> List[Tuple[str, str, Path]]:
    """Find PatchCore checkpoints.

    Expected structure:
        checkpoint_dir/Patchcore/
            GoodsAD/
                cigarette_box/
                    v0/model.ckpt

    Args:
        checkpoint_dir: Root checkpoint directory
        version: Specific version to use (None = latest with model.ckpt)
        verbose: Print debug output

    Returns:
        List of (dataset, category, checkpoint_path)
    """
    checkpoints = []
    patchcore_dir = checkpoint_dir / "Patchcore"

    if not patchcore_dir.exists():
        # Try without Patchcore subdirectory
        patchcore_dir = checkpoint_dir

    if verbose:
        print(f"[DEBUG] Searching in: {patchcore_dir}")
        print(f"[DEBUG] Version: {version if version is not None else 'auto (latest)'}")

    for dataset_dir in sorted(patchcore_dir.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
            continue
        if dataset_dir.name in ["eval", "predictions", "Patchcore"]:
            continue

        if verbose:
            print(f"[DEBUG] Dataset: {dataset_dir.name}")

        for category_dir in sorted(dataset_dir.iterdir()):
            if not category_dir.is_dir() or category_dir.name.startswith("."):
                continue

            # If specific version requested, use it strictly (no fallback)
            if version is not None:
                ver_dir = category_dir / f"v{version}"
                ckpt_path = ver_dir / "model.ckpt"
                if ckpt_path.exists():
                    checkpoints.append((dataset_dir.name, category_dir.name, ckpt_path))
                    if verbose:
                        print(f"[DEBUG]   {category_dir.name}: v{version} ✓")
                elif verbose:
                    print(f"[DEBUG]   {category_dir.name}: v{version} ✗ (model.ckpt not found)")
                continue

            # Auto mode (version=None): find latest version with model.ckpt
            versions = []
            for v_dir in category_dir.iterdir():
                if v_dir.is_dir() and v_dir.name.startswith("v"):
                    try:
                        versions.append((int(v_dir.name[1:]), v_dir))
                    except ValueError:
                        continue

            if not versions:
                continue

            versions_sorted = sorted(versions, key=lambda x: x[0], reverse=True)
            for ver_num, ver_dir in versions_sorted:
                candidate = ver_dir / "model.ckpt"
                if candidate.exists():
                    checkpoints.append((dataset_dir.name, category_dir.name, candidate))
                    if verbose:
                        print(f"[DEBUG]   {category_dir.name}: v{ver_num} ✓")
                    break
            else:
                if verbose:
                    print(f"[DEBUG]   {category_dir.name}: no model.ckpt found")

    return checkpoints


class PatchCoreONNXWrapper(torch.nn.Module):
    """Wrapper for PatchCore model that includes memory bank for ONNX export."""

    def __init__(self, model, input_size: Tuple[int, int]):
        super().__init__()
        self.model = model
        self.input_size = input_size

        # Get the feature extractor and memory bank from anomalib model
        if hasattr(model, "model"):
            inner_model = model.model
        else:
            inner_model = model

        # Copy feature extractor
        self.feature_extractor = inner_model.feature_extractor
        self.feature_pooler = inner_model.feature_pooler

        # Register memory bank as buffer (will be included in ONNX)
        memory_bank = inner_model.memory_bank.clone()
        self.register_buffer("memory_bank", memory_bank)

        # Get anomaly map generator settings
        self.anomaly_map_generator = inner_model.anomaly_map_generator

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass that computes anomaly map and score.

        Args:
            x: Input image tensor [B, 3, H, W]

        Returns:
            anomaly_map: [B, 1, H, W]
            pred_score: [B]
        """
        batch_size = x.shape[0]

        # Extract features
        features = self.feature_extractor(x)
        features = self.feature_pooler(features)

        # Reshape features for distance computation
        # features: [B, C, H, W] -> [B*H*W, C]
        b, c, h, w = features.shape
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, c)

        # Compute distances to memory bank
        # memory_bank: [N, C]
        # distances: [B*H*W, N]
        distances = torch.cdist(features_flat, self.memory_bank, p=2)

        # Get minimum distance for each patch
        min_distances, _ = distances.min(dim=1)  # [B*H*W]

        # Reshape to anomaly map
        anomaly_map = min_distances.reshape(b, 1, h, w)

        # Upsample to input size
        anomaly_map = torch.nn.functional.interpolate(
            anomaly_map,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
        )

        # Compute anomaly score (max of anomaly map)
        pred_score = anomaly_map.reshape(batch_size, -1).max(dim=1)[0]

        return anomaly_map, pred_score


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    input_size: Tuple[int, int] = (224, 224),
    opset_version: int = 14,
) -> bool:
    """Export PatchCore checkpoint to ONNX with memory bank included.

    Args:
        checkpoint_path: Path to .ckpt file
        output_path: Path to save .onnx file
        input_size: Model input size (height, width)
        opset_version: ONNX opset version

    Returns:
        True if successful, False otherwise
    """
    try:
        from anomalib.models import Patchcore

        print(f"  Loading checkpoint: {checkpoint_path}")
        model = Patchcore.load_from_checkpoint(str(checkpoint_path), map_location="cpu")
        model.eval()
        model = model.cpu()

        # Check memory bank size
        if hasattr(model, "model") and hasattr(model.model, "memory_bank"):
            memory_bank = model.model.memory_bank
            print(f"  Memory bank shape: {memory_bank.shape}")
            memory_mb = memory_bank.numel() * 4 / 1024 / 1024  # float32 = 4 bytes
            print(f"  Memory bank size: {memory_mb:.1f} MB")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Exporting to ONNX: {output_path}")

        # Create wrapper model with memory bank
        try:
            wrapper = PatchCoreONNXWrapper(model, input_size)
            wrapper.eval()

            dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

            # Test forward pass
            with torch.no_grad():
                anomaly_map, pred_score = wrapper(dummy_input)
                print(f"  Test output - anomaly_map: {anomaly_map.shape}, pred_score: {pred_score.shape}")

            # Export to ONNX
            torch.onnx.export(
                wrapper,
                dummy_input,
                str(output_path),
                opset_version=opset_version,
                input_names=["input"],
                output_names=["anomaly_map", "pred_score"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "anomaly_map": {0: "batch_size"},
                    "pred_score": {0: "batch_size"},
                },
            )

            # Verify file size (should be > 1MB if memory bank included)
            file_size_mb = output_path.stat().st_size / 1024 / 1024
            print(f"  ONNX file size: {file_size_mb:.1f} MB")

            if file_size_mb < 1:
                print(f"  Warning: File size too small, memory bank may not be included")

            # Verify the exported model
            try:
                import onnx
                onnx_model = onnx.load(str(output_path))
                onnx.checker.check_model(onnx_model)
                print(f"  Success! Model verified.")
            except ImportError:
                print(f"  Success! (onnx package not available for verification)")
            except Exception as e:
                print(f"  Warning: Model verification failed: {e}")

            return True

        except Exception as e:
            print(f"  Wrapper export failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Export PatchCore checkpoints to ONNX")

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="output",
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/onnx",
        help="Directory to save ONNX models",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config file with datasets and categories to export",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Export only specific dataset",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Export only specific category",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Input size (height width)",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list available checkpoints",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing ONNX files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show debug output for checkpoint discovery",
    )

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)

    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    # Load config if provided
    config_datasets = None
    config_categories = None
    config_version = None
    if args.config:
        from src.utils.loaders import load_config
        config = load_config(args.config)
        config_datasets = config.get("data", {}).get("datasets", None)
        config_categories = config.get("data", {}).get("categories", None)
        config_version = config.get("predict", {}).get("version", None)
        print(f"Using config: {args.config}")
        print(f"  Datasets: {config_datasets}")
        print(f"  Categories: {config_categories}")
        print(f"  Version: {config_version if config_version is not None else 'auto (latest)'}")
        print()

    # Find checkpoints
    checkpoints = find_patchcore_checkpoints(checkpoint_dir, version=config_version, verbose=args.verbose)

    if not checkpoints:
        print(f"No PatchCore checkpoints found in {checkpoint_dir}")
        sys.exit(1)

    # Filter by config
    if config_datasets:
        checkpoints = [c for c in checkpoints if c[0] in config_datasets]
    if config_categories:
        checkpoints = [c for c in checkpoints if c[1] in config_categories]

    # Filter by command line args (override config)
    if args.dataset:
        checkpoints = [c for c in checkpoints if c[0] == args.dataset]
    if args.category:
        checkpoints = [c for c in checkpoints if c[1] == args.category]

    print(f"Found {len(checkpoints)} checkpoint(s):")
    for dataset, category, ckpt_path in checkpoints:
        print(f"  [{dataset}] {category}: {ckpt_path}")

    # Warn about missing checkpoints from config
    if config_categories:
        found_categories = {c[1] for c in checkpoints}
        missing = set(config_categories) - found_categories
        if missing:
            print(f"\nWarning: Missing checkpoints for categories: {sorted(missing)}")

    if args.list_only:
        return

    print()
    print("=" * 60)
    print("Exporting to ONNX")
    print("=" * 60)

    success_count = 0
    skip_count = 0
    fail_count = 0

    for dataset, category, ckpt_path in checkpoints:
        print(f"\n[{dataset}] {category}")

        output_path = output_dir / dataset / category / "model.onnx"

        if output_path.exists() and not args.force:
            print(f"  Skipping (already exists): {output_path}")
            skip_count += 1
            continue

        if export_to_onnx(
            ckpt_path,
            output_path,
            input_size=tuple(args.input_size),
            opset_version=args.opset_version,
        ):
            success_count += 1
        else:
            fail_count += 1

    print()
    print("=" * 60)
    print(f"Export complete: {success_count} success, {skip_count} skipped, {fail_count} failed")
    print(f"ONNX models saved to: {output_dir}")


if __name__ == "__main__":
    main()
