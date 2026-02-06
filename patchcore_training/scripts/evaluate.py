#!/usr/bin/env python
"""Evaluate PatchCore models.

Usage:
    # Evaluate all trained models
    python patchcore_training/scripts/evaluate.py

    # Evaluate with custom config
    python patchcore_training/scripts/evaluate.py --config patchcore_training/config/config.yaml

    # Evaluate specific category
    python patchcore_training/scripts/evaluate.py --dataset GoodsAD --category cigarette_box

    # Save results to file
    python patchcore_training/scripts/evaluate.py --output output/patchcore_eval_results.json
"""

import argparse
import sys
from pathlib import Path

# Add patchcore_training to path
SCRIPT_DIR = Path(__file__).resolve().parent
PATCHCORE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PATCHCORE_ROOT))

from src.trainer import PatchCoreTrainer
from src.evaluator import PatchCoreEvaluator
from src.utils import load_config, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PatchCore models")

    parser.add_argument(
        "--config",
        type=str,
        default=str(PATCHCORE_ROOT / "config" / "config.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Evaluate only specific dataset",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Evaluate only specific category (requires --dataset)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="Output path for per-sample predictions CSV",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Path to per-category thresholds YAML file (e.g., config/thresholds.yaml)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    setup_seed(args.seed)

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Load per-category thresholds if provided
    per_category_thresholds = None
    if args.thresholds:
        import yaml
        from pathlib import Path
        thresholds_path = Path(args.thresholds)
        if thresholds_path.exists():
            with open(thresholds_path, "r", encoding="utf-8") as f:
                thresholds_config = yaml.safe_load(f)
            global_threshold = thresholds_config.get("global", 0.5)
            per_category_thresholds = thresholds_config.get("categories", {})
            # Update evaluator's default threshold
            config.setdefault("evaluation", {})["threshold"] = global_threshold
            print(f"Loaded per-category thresholds from: {thresholds_path}")
            print(f"  Global fallback: {global_threshold}")
            print(f"  Categories: {len(per_category_thresholds)}")
        else:
            print(f"Warning: Thresholds file not found: {thresholds_path}")

    # Create trainer and evaluator
    trainer = PatchCoreTrainer(config)
    evaluator = PatchCoreEvaluator(config)

    # Evaluate
    if args.dataset and args.category:
        # Evaluate single category
        category_key = f"{args.dataset}/{args.category}"
        print(f"\nEvaluating: {category_key}")
        model = trainer.load_model(args.dataset, args.category)
        if model is None:
            print(f"Model not found for {category_key}")
            return

        # Get category-specific threshold if available
        category_threshold = None
        if per_category_thresholds:
            category_threshold = per_category_thresholds.get(category_key)

        metrics = evaluator.evaluate_category(
            model=model,
            dataset_name=args.dataset,
            category=args.category,
            threshold=category_threshold,
        )
        results = {category_key: metrics}
    else:
        # Load all models
        models = trainer.load_all_models()
        if not models:
            print("No trained models found!")
            return

        # Filter by dataset if specified
        if args.dataset:
            models = {k: v for k, v in models.items() if k.startswith(args.dataset + "/")}

        # Evaluate all
        results = evaluator.evaluate_all(
            models,
            save_predictions_csv=args.save_csv,
            per_category_thresholds=per_category_thresholds,
        )

    # Save results if output path specified
    if args.output:
        evaluator.save_results(results, args.output)
    else:
        # Print summary
        print("\n" + "=" * 60)
        print("Evaluation Results Summary")
        print("=" * 60)
        for key, metrics in results.items():
            if key == "average":
                continue
            print(f"\n{key}:")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric_name}: {value:.4f}")
                else:
                    print(f"  {metric_name}: {value}")

    print("\nDone!")


if __name__ == "__main__":
    main()
