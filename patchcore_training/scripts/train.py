#!/usr/bin/env python
"""Train PatchCore models.

Usage:
    # Train all categories in config
    python patchcore_training/scripts/train.py

    # Train with custom config
    python patchcore_training/scripts/train.py --config patchcore_training/config/config.yaml

    # Train specific category only
    python patchcore_training/scripts/train.py --dataset GoodsAD --category cigarette_box
"""

import argparse
import sys
from pathlib import Path

# Add patchcore_training to path
SCRIPT_DIR = Path(__file__).resolve().parent
PATCHCORE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PATCHCORE_ROOT))

from src.trainer import PatchCoreTrainer
from src.utils import load_config, setup_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train PatchCore models")

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
        help="Train only specific dataset",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Train only specific category (requires --dataset)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    setup_seed(args.seed)

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Create trainer
    trainer = PatchCoreTrainer(config)

    # Train
    if args.dataset and args.category:
        # Train single category
        print(f"\nTraining single category: {args.dataset}/{args.category}")
        model = trainer.train_category(args.dataset, args.category)
        if model:
            print("\nTraining complete!")
    elif args.dataset:
        # Train all categories in dataset
        print(f"\nTraining all categories in dataset: {args.dataset}")
        datasets_config = config["data"].get("datasets", {})
        if args.dataset in datasets_config:
            for category in datasets_config[args.dataset]:
                trainer.train_category(args.dataset, category)
        else:
            print(f"Dataset {args.dataset} not found in config")
    else:
        # Train all
        trainer.train_all()

    print("\nDone!")


if __name__ == "__main__":
    main()
