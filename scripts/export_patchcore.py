"""Export PatchCore models for optimized inference.

Strategy: Export backbone to ONNX + save memory bank separately.
This avoids anomalib's ONNX export issues while enabling fast inference.

Usage:
    python scripts/export_patchcore.py \
        --checkpoint-dir /path/to/checkpoints \
        --output-dir models/onnx \
        --config configs/anomaly.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

SCRIPT_PATH = Path(__file__).resolve()
PROJ_ROOT = SCRIPT_PATH.parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))


class BackboneWrapper(nn.Module):
    """Wrapper that extracts and concatenates features from backbone."""

    def __init__(self, feature_extractor, layers: Tuple[str, ...]):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and concatenate features from specified layers."""
        features_dict = self.feature_extractor(x)

        # Get target size from first layer
        first_layer = self.layers[0]
        target_size = features_dict[first_layer].shape[-2:]

        # Concatenate features from all layers
        features_list = []
        for layer in self.layers:
            feat = features_dict[layer]
            if feat.shape[-2:] != target_size:
                feat = nn.functional.interpolate(
                    feat, size=target_size, mode="bilinear", align_corners=False
                )
            features_list.append(feat)

        return torch.cat(features_list, dim=1)


def find_checkpoints(
    checkpoint_dir: Path,
    version: int | None = None,
    datasets: List[str] | None = None,
    categories: List[str] | None = None,
) -> List[Tuple[str, str, Path]]:
    """Find PatchCore checkpoints."""
    checkpoints = []
    patchcore_dir = checkpoint_dir / "Patchcore"

    if not patchcore_dir.exists():
        patchcore_dir = checkpoint_dir

    for dataset_dir in sorted(patchcore_dir.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
            continue
        if dataset_dir.name in ["eval", "predictions", "Patchcore"]:
            continue
        if datasets and dataset_dir.name not in datasets:
            continue

        for category_dir in sorted(dataset_dir.iterdir()):
            if not category_dir.is_dir() or category_dir.name.startswith("."):
                continue
            if categories and category_dir.name not in categories:
                continue

            # Find checkpoint
            if version is not None:
                # Strict version: only use specified version, no fallback
                ckpt = category_dir / f"v{version}" / "model.ckpt"
                if ckpt.exists():
                    checkpoints.append((dataset_dir.name, category_dir.name, ckpt))
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
                    ckpt = latest / "model.ckpt"
                    if ckpt.exists():
                        checkpoints.append((dataset_dir.name, category_dir.name, ckpt))

    return checkpoints


def export_model(
    checkpoint_path: Path,
    output_dir: Path,
    input_size: Tuple[int, int] = (700, 700),
) -> bool:
    """Export PatchCore model: backbone to ONNX + memory bank to numpy.

    Output files:
        - backbone.onnx: Feature extractor
        - memory_bank.npy: Coreset embeddings
        - config.json: Model configuration
    """
    try:
        from anomalib.models import Patchcore

        print(f"  Loading: {checkpoint_path}")
        model = Patchcore.load_from_checkpoint(str(checkpoint_path), map_location="cpu")
        model.eval()

        inner = model.model
        memory_bank = inner.memory_bank.cpu().numpy()
        layers = inner.layers

        print(f"  Memory bank: {memory_bank.shape} ({memory_bank.nbytes / 1024 / 1024:.1f} MB)")
        print(f"  Layers: {layers}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save memory bank
        memory_bank_path = output_dir / "memory_bank.npy"
        np.save(memory_bank_path, memory_bank)
        print(f"  Saved: {memory_bank_path}")

        # 2. Export backbone to ONNX
        backbone_path = output_dir / "backbone.onnx"
        backbone = BackboneWrapper(inner.feature_extractor, layers)
        backbone.eval()

        dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

        # Test forward
        with torch.no_grad():
            test_output = backbone(dummy_input)
            print(f"  Backbone output: {test_output.shape}")

        # Export with legacy mode
        torch.onnx.export(
            backbone,
            dummy_input,
            str(backbone_path),
            opset_version=14,
            input_names=["input"],
            output_names=["features"],
            dynamic_axes={
                "input": {0: "batch"},
                "features": {0: "batch"},
            },
        )

        backbone_size = backbone_path.stat().st_size / 1024 / 1024
        print(f"  Saved: {backbone_path} ({backbone_size:.1f} MB)")

        # 3. Save config
        import json
        config = {
            "input_size": list(input_size),
            "layers": list(layers),
            "feature_dim": int(test_output.shape[1]),
            "feature_map_size": [int(test_output.shape[2]), int(test_output.shape[3])],
            "memory_bank_size": int(memory_bank.shape[0]),
        }
        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Saved: {config_path}")

        # 4. Verify ONNX
        try:
            import onnx
            onnx_model = onnx.load(str(backbone_path))
            onnx.checker.check_model(onnx_model)
            print(f"  ONNX verified!")
        except Exception as e:
            print(f"  ONNX verification warning: {e}")

        return True

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Export PatchCore for optimized inference")
    parser.add_argument("--checkpoint-dir", type=str, default="output")
    parser.add_argument("--output-dir", type=str, default="models/onnx")
    parser.add_argument("--config", type=str, default="configs/anomaly.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--input-size", type=int, nargs=2, default=[700, 700])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)

    # Load config
    config_version = None
    config_datasets = None
    config_categories = None
    input_size = tuple(args.input_size)

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
        print(f"  Datasets: {config_datasets}")
        print(f"  Categories: {config_categories}")
        print(f"  Version: v{config_version}" if config_version else "  Version: latest")
        print(f"  Input size: {input_size}")
        print()

    # Override with CLI args
    if args.dataset:
        config_datasets = [args.dataset]
    if args.category:
        config_categories = [args.category]

    # Find checkpoints
    checkpoints = find_checkpoints(
        checkpoint_dir,
        version=config_version,
        datasets=config_datasets,
        categories=config_categories,
    )

    if not checkpoints:
        print("No checkpoints found!")
        return

    print(f"Found {len(checkpoints)} checkpoint(s)")
    for dataset, category, path in checkpoints:
        print(f"  [{dataset}] {category}: {path}")

    if args.list_only:
        return

    # Export
    print()
    print("=" * 60)
    print("Exporting models")
    print("=" * 60)

    success = 0
    failed = 0
    skipped = 0

    for dataset, category, ckpt_path in checkpoints:
        print(f"\n[{dataset}/{category}]")

        model_output_dir = output_dir / dataset / category

        # Check if already exists
        if not args.force and (model_output_dir / "backbone.onnx").exists():
            print(f"  Skipped (exists)")
            skipped += 1
            continue

        if export_model(ckpt_path, model_output_dir, input_size):
            success += 1
        else:
            failed += 1

    print()
    print("=" * 60)
    print(f"Complete: {success} success, {skipped} skipped, {failed} failed")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
