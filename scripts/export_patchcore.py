"""Export PatchCore checkpoints to ONNX format.

Usage:
    # Export all trained models
    python scripts/export_patchcore.py --checkpoint-dir output --output-dir models/onnx

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


def find_patchcore_checkpoints(checkpoint_dir: Path) -> List[Tuple[str, str, Path]]:
    """Find PatchCore checkpoints.

    Expected structure:
        checkpoint_dir/Patchcore/
            GoodsAD/
                cigarette_box/
                    v0/model.ckpt

    Returns:
        List of (dataset, category, checkpoint_path)
    """
    checkpoints = []
    patchcore_dir = checkpoint_dir / "Patchcore"

    if not patchcore_dir.exists():
        # Try without Patchcore subdirectory
        patchcore_dir = checkpoint_dir

    for dataset_dir in patchcore_dir.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
            continue
        if dataset_dir.name in ["eval", "predictions", "Patchcore"]:
            continue

        for category_dir in dataset_dir.iterdir():
            if not category_dir.is_dir() or category_dir.name.startswith("."):
                continue

            # Find latest version
            versions = []
            for v_dir in category_dir.iterdir():
                if v_dir.is_dir() and v_dir.name.startswith("v"):
                    try:
                        versions.append((int(v_dir.name[1:]), v_dir))
                    except ValueError:
                        continue

            if not versions:
                continue

            latest_version_dir = max(versions, key=lambda x: x[0])[1]
            ckpt_path = latest_version_dir / "model.ckpt"

            if ckpt_path.exists():
                checkpoints.append((dataset_dir.name, category_dir.name, ckpt_path))

    return checkpoints


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    input_size: Tuple[int, int] = (224, 224),
    opset_version: int = 14,
) -> bool:
    """Export PatchCore checkpoint to ONNX.

    Uses anomalib v2.0+ Engine.export() API with fallback to manual export.

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
        model = Patchcore.load_from_checkpoint(str(checkpoint_path))
        model.eval()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"  Exporting to ONNX: {output_path}")

        exported = False

        # Method 1: Use Engine.export() (anomalib v2.0+ recommended)
        try:
            from anomalib.engine import Engine

            engine = Engine()
            engine.export(
                model=model,
                export_type="onnx",
                export_root=str(output_path.parent),
                input_size=input_size,
                ckpt_path=str(checkpoint_path),
            )
            # Engine exports to export_root/weights/onnx/model.onnx
            engine_output = output_path.parent / "weights" / "onnx" / "model.onnx"
            if engine_output.exists():
                # Move to our desired location
                import shutil
                shutil.move(str(engine_output), str(output_path))
                # Clean up engine's directory structure
                shutil.rmtree(output_path.parent / "weights", ignore_errors=True)
                exported = True
                print(f"  Exported via Engine.export()")
        except Exception as e:
            print(f"  Engine.export() failed: {e}")

        # Method 2: Use model.to_onnx() (anomalib v1.x compatible)
        if not exported and hasattr(model, "to_onnx"):
            try:
                model.to_onnx(
                    export_path=output_path,
                    input_size=input_size,
                )
                exported = True
                print(f"  Exported via model.to_onnx()")
            except Exception as e:
                print(f"  model.to_onnx() failed: {e}")

        # Method 3: Manual torch.onnx.export (fallback)
        if not exported:
            try:
                dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

                # Get the torch model
                if hasattr(model, "model"):
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
                    output_names=["anomaly_map", "pred_score"],
                    dynamic_axes={
                        "input": {0: "batch_size"},
                        "anomaly_map": {0: "batch_size"},
                        "pred_score": {0: "batch_size"},
                    },
                )
                exported = True
                print(f"  Exported via torch.onnx.export()")
            except Exception as e:
                print(f"  Manual export failed: {e}")

        if exported:
            # Verify the exported model
            try:
                import onnx
                onnx_model = onnx.load(str(output_path))
                onnx.checker.check_model(onnx_model)
                print(f"  Success! Model verified.")
                return True
            except ImportError:
                print(f"  Success! (onnx package not available for verification)")
                return True
            except Exception as e:
                print(f"  Warning: Model verification failed: {e}")
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

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)

    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    # Find checkpoints
    checkpoints = find_patchcore_checkpoints(checkpoint_dir)

    if not checkpoints:
        print(f"No PatchCore checkpoints found in {checkpoint_dir}")
        sys.exit(1)

    # Filter by dataset/category
    if args.dataset:
        checkpoints = [c for c in checkpoints if c[0] == args.dataset]
    if args.category:
        checkpoints = [c for c in checkpoints if c[1] == args.category]

    print(f"Found {len(checkpoints)} checkpoint(s):")
    for dataset, category, ckpt_path in checkpoints:
        print(f"  [{dataset}] {category}: {ckpt_path}")

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
