"""Convert anomalib checkpoints to ONNX format.

Supports:
- EfficientAD (per-class models)
- PatchCore (per-class models) - planned
- UniAD (unified model) - planned

Usage:
    # Convert all EfficientAD models in output directory
    python scripts/convert_to_onnx.py --model-type efficientad --checkpoint-dir output --output-dir models/onnx

    # Convert specific class
    python scripts/convert_to_onnx.py --model-type efficientad --checkpoint-dir output --output-dir models/onnx --class-name cigarette_box

    # List available checkpoints
    python scripts/convert_to_onnx.py --model-type efficientad --checkpoint-dir output --list-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Add project root to path
SCRIPT_PATH = Path(__file__).resolve()
PROJ_ROOT = SCRIPT_PATH.parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))


def find_efficientad_checkpoints(checkpoint_dir: Path) -> List[Tuple[str, str, Path]]:
    """Find EfficientAD checkpoints in anomalib output format.

    Expected structure:
        checkpoint_dir/
            GoodsAD/
                cigarette_box/
                    v0/weights/lightning/model.ckpt
            MVTec-LOCO/
                breakfast_box/
                    v0/weights/lightning/model.ckpt

    Returns:
        List of (dataset, class_name, checkpoint_path)
    """
    checkpoints = []

    for dataset_dir in checkpoint_dir.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
            continue
        if dataset_dir.name == "eval":  # Skip eval directory
            continue

        for class_dir in dataset_dir.iterdir():
            if not class_dir.is_dir() or class_dir.name.startswith("."):
                continue

            # Find checkpoint file
            ckpt_patterns = [
                class_dir / "v0" / "weights" / "lightning" / "model.ckpt",
                class_dir / "weights" / "lightning" / "model.ckpt",
                class_dir / "model.ckpt",
            ]

            for ckpt_path in ckpt_patterns:
                if ckpt_path.exists():
                    checkpoints.append((dataset_dir.name, class_dir.name, ckpt_path))
                    break

    return checkpoints


def convert_efficientad_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    input_size: Tuple[int, int] = (256, 256),
    opset_version: int = 14,
) -> bool:
    """Convert EfficientAD checkpoint to ONNX.

    Args:
        checkpoint_path: Path to .ckpt file
        output_path: Path to save .onnx file
        input_size: Model input size (height, width)
        opset_version: ONNX opset version

    Returns:
        True if successful, False otherwise
    """
    try:
        import torch
    except ImportError as e:
        print(f"Error: torch not installed: {e}")
        print("Install with: pip install torch")
        return False

    try:
        from anomalib.models import EfficientAd
    except ImportError as e:
        print(f"Error: anomalib not installed: {e}")
        print("Install with: pip install anomalib")
        return False

    try:
        # Load model from checkpoint
        print(f"  Loading checkpoint: {checkpoint_path}")
        model = EfficientAd.load_from_checkpoint(checkpoint_path)
        model.eval()

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export to ONNX
        print(f"  Exporting to ONNX: {output_path}")

        # Create dummy input
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

        # Try using anomalib's export methods (API varies by version)
        exported = False

        # Try anomalib 1.x API
        if hasattr(model, 'to_onnx'):
            try:
                model.to_onnx(
                    export_path=output_path,
                    input_size=input_size,
                )
                exported = True
            except Exception:
                pass

        # Try anomalib 0.x API or direct export
        if not exported:
            try:
                # Get the actual model for inference
                if hasattr(model, 'model'):
                    export_model = model.model
                else:
                    export_model = model

                export_model.eval()

                torch.onnx.export(
                    export_model,
                    dummy_input,
                    str(output_path),
                    opset_version=opset_version,
                    input_names=["input"],
                    output_names=["anomaly_map"],
                    dynamic_axes={
                        "input": {0: "batch_size"},
                        "anomaly_map": {0: "batch_size"},
                    },
                )
                exported = True
            except Exception as e2:
                print(f"  Warning: Direct export failed: {e2}")

        if exported:
            print(f"  Success!")
            return True
        else:
            print(f"  Failed to export")
            return False

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert anomalib checkpoints to ONNX")

    parser.add_argument(
        "--model-type",
        type=str,
        default="efficientad",
        choices=["efficientad", "patchcore", "uniad"],
        help="Type of anomaly detection model",
    )
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
        "--class-name",
        type=str,
        default=None,
        help="Convert only specific class (optional)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Convert only specific dataset (optional)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[256, 256],
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

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)

    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    # Find checkpoints
    if args.model_type == "efficientad":
        checkpoints = find_efficientad_checkpoints(checkpoint_dir)
    else:
        print(f"Error: Model type '{args.model_type}' not yet supported")
        sys.exit(1)

    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        sys.exit(1)

    # Filter by dataset/class if specified
    if args.dataset:
        checkpoints = [c for c in checkpoints if c[0] == args.dataset]
    if args.class_name:
        checkpoints = [c for c in checkpoints if c[1] == args.class_name]

    print(f"Found {len(checkpoints)} checkpoint(s):")
    for dataset, class_name, ckpt_path in checkpoints:
        print(f"  [{dataset}] {class_name}: {ckpt_path}")

    if args.list_only:
        return

    print()
    print("=" * 60)
    print("Converting to ONNX")
    print("=" * 60)

    success_count = 0
    fail_count = 0

    for dataset, class_name, ckpt_path in checkpoints:
        print(f"\n[{dataset}] {class_name}")

        # Output path: models/onnx/{dataset}/{class_name}/model.onnx
        output_path = output_dir / dataset / class_name / "model.onnx"

        if output_path.exists():
            print(f"  Skipping (already exists): {output_path}")
            success_count += 1
            continue

        if convert_efficientad_to_onnx(
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
    print(f"Conversion complete: {success_count} success, {fail_count} failed")
    print(f"ONNX models saved to: {output_dir}")


if __name__ == "__main__":
    main()
