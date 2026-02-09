"""Experiment Runner — run MMAD evaluation with YAML config + CLI overrides.

Usage:
    # Use YAML config
    python scripts/run_experiment.py

    # CLI overrides
    python scripts/run_experiment.py --llm gpt-4o --ad-model null
    python scripts/run_experiment.py --llm qwen --ad-model patchcore
    python scripts/run_experiment.py --llm qwen --max-images 5

    # List available models
    python scripts/run_experiment.py --list-models

    # Custom config file
    python scripts/run_experiment.py --config configs/experiment.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# Path setup
SCRIPT_PATH = Path(__file__).resolve()
PROJ_ROOT = SCRIPT_PATH.parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.config.experiment import ExperimentConfig, load_experiment_config
from src.mllm.factory import MODEL_REGISTRY, get_llm_client, list_llm_models
from src.eval.metrics import calculate_accuracy_mmad


def load_mmad_data(json_path: str) -> dict:
    """Load MMAD dataset JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_paths(cfg: ExperimentConfig) -> tuple[str, str]:
    """Resolve data_root and mmad_json from config, env vars, or defaults."""
    data_root = cfg.data_root or os.environ.get("MMAD_DATA_ROOT")
    mmad_json = cfg.mmad_json or os.environ.get("MMAD_JSON_PATH")

    if not data_root:
        candidates = [
            PROJ_ROOT / "datasets" / "MMAD",
            Path("/Users/leehw/Documents/likelion/final_project/MMAD/dataset/MMAD"),
        ]
        for c in candidates:
            if c.exists():
                data_root = str(c)
                break

    if not mmad_json and data_root:
        mmad_json = str(Path(data_root) / "mmad.json")

    return data_root, mmad_json


def run_experiment(cfg: ExperimentConfig) -> Path:
    """Run a single experiment and return the answers JSON path."""
    data_root, mmad_json = resolve_paths(cfg)

    # Validate paths
    if not mmad_json or not Path(mmad_json).exists():
        print(f"Error: mmad.json not found at {mmad_json}")
        print("Set --data-root, MMAD_DATA_ROOT, or data_root in YAML")
        sys.exit(1)

    if not data_root or not Path(data_root).exists():
        print(f"Error: Data root not found at {data_root}")
        print("Set --data-root, MMAD_DATA_ROOT, or data_root in YAML")
        sys.exit(1)

    # Setup output
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    template_type = "Similar_template" if cfg.similar_template else "Random_template"
    ad_suffix = f"_with_{cfg.ad_model}" if cfg.ad_model else ""
    llm_safe = cfg.llm.replace("/", "_").replace("\\", "_")
    output_name = f"answers_{cfg.few_shot}_shot_{llm_safe}_{template_type}{ad_suffix}"
    answers_json_path = output_dir / f"{output_name}.json"

    print("=" * 60)
    print("MMAD Experiment Runner")
    print("=" * 60)
    print(f"Experiment:  {cfg.experiment_name}")
    print(f"LLM:         {cfg.llm}")
    print(f"AD model:    {cfg.ad_model or 'none'}")
    print(f"Few-shot:    {cfg.few_shot}")
    print(f"Template:    {template_type}")
    print(f"Image size:  {cfg.max_image_size}")
    print(f"Max images:  {cfg.max_images or 'all'}")
    print(f"Data root:   {data_root}")
    print(f"Output:      {answers_json_path}")
    print("=" * 60)
    print()

    # Load existing results if resuming
    all_answers = []
    existing_images = set()

    if cfg.resume and answers_json_path.exists():
        with open(answers_json_path, "r", encoding="utf-8") as f:
            all_answers = json.load(f)
        existing_images = {a["image"] for a in all_answers}
        print(f"Resuming from {len(existing_images)} existing images")

    # Initialize LLM client
    try:
        llm_client = get_llm_client(cfg.llm, max_image_size=cfg.max_image_size)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Load dataset
    mmad_data = load_mmad_data(mmad_json)
    image_paths = list(mmad_data.keys())

    if cfg.max_images:
        image_paths = image_paths[:cfg.max_images]

    print(f"Total images: {len(image_paths)}")
    print()

    # Track statistics
    total_correct = 0
    total_questions = 0
    processed = 0
    errors = 0
    start_time = time.time()

    # Evaluate with progress bar
    pbar = tqdm(image_paths, desc="Evaluating", ncols=100)
    for image_rel in pbar:
        if image_rel in existing_images:
            continue

        meta = mmad_data[image_rel]

        # Get templates
        if cfg.similar_template:
            few_shot = meta.get("similar_templates", [])[:cfg.few_shot]
        else:
            few_shot = meta.get("random_templates", [])[:cfg.few_shot]

        # Build absolute paths
        query_image_path = str(Path(data_root) / image_rel)
        few_shot_paths = [str(Path(data_root) / p) for p in few_shot]

        # Check if image exists
        if not Path(query_image_path).exists():
            errors += 1
            continue

        # Generate answers
        if cfg.batch_mode:
            questions, answers, predicted, q_types = llm_client.generate_answers_batch(
                query_image_path, meta, few_shot_paths
            )
        else:
            questions, answers, predicted, q_types = llm_client.generate_answers(
                query_image_path, meta, few_shot_paths
            )

        if predicted is None or len(predicted) != len(answers):
            errors += 1
            continue

        # Calculate accuracy for this image
        correct = sum(1 for p, a in zip(predicted, answers) if p == a)
        total_correct += correct
        total_questions += len(answers)
        processed += 1

        # Update progress bar with running accuracy
        running_acc = total_correct / total_questions if total_questions > 0 else 0
        pbar.set_postfix({"acc": f"{running_acc:.1%}", "done": processed, "err": errors}, refresh=False)

        # Store results
        for q, a, pred, qt in zip(questions, answers, predicted, q_types):
            all_answers.append({
                "image": image_rel,
                "question": q,
                "question_type": qt,
                "correct_answer": a,
                "gpt_answer": pred,
            })

        # Save incrementally
        with open(answers_json_path, "w", encoding="utf-8") as f:
            json.dump(all_answers, f, indent=4, ensure_ascii=False)

    pbar.close()

    elapsed = time.time() - start_time
    final_acc = total_correct / total_questions if total_questions > 0 else 0

    # Save metadata
    meta_path = answers_json_path.with_suffix(".meta.json")
    meta_info = {
        "experiment_name": cfg.experiment_name,
        "llm": cfg.llm,
        "ad_model": cfg.ad_model,
        "few_shot": cfg.few_shot,
        "similar_template": cfg.similar_template,
        "max_images": cfg.max_images,
        "total_images": len(image_paths),
        "processed": processed,
        "errors": errors,
        "total_questions": total_questions,
        "total_correct": total_correct,
        "accuracy": round(final_acc * 100, 2),
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
        "answers_file": str(answers_json_path),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2, ensure_ascii=False)

    # Print summary
    print()
    print("=" * 60)
    print("Calculating Metrics")
    print("=" * 60)
    if total_questions > 0:
        calculate_accuracy_mmad(str(answers_json_path))
    else:
        print("No valid answers to evaluate")
    print()
    print(f"Results saved to: {answers_json_path}")
    print(f"Metadata saved to: {meta_path}")
    print(f"Elapsed: {elapsed:.1f}s")

    return answers_json_path


def main():
    parser = argparse.ArgumentParser(description="MMAD Experiment Runner")

    parser.add_argument("--config", type=str, default="configs/experiment.yaml",
                        help="Experiment config YAML (default: configs/experiment.yaml)")

    # CLI overrides
    parser.add_argument("--llm", type=str, default=None,
                        help="Override LLM model name")
    parser.add_argument("--ad-model", type=str, default=None,
                        help="Override AD model (null = no AD)")
    parser.add_argument("--few-shot", type=int, default=None,
                        help="Override few-shot count")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Override max images")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override data root path")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--resume", action="store_true", default=None,
                        help="Resume from existing results")

    # Utility
    parser.add_argument("--list-models", action="store_true",
                        help="List available LLM models and exit")

    args = parser.parse_args()

    # List models mode
    if args.list_models:
        print("Available LLM models:")
        print("-" * 40)
        for name in list_llm_models():
            info = MODEL_REGISTRY[name]
            print(f"  {name:<20s} ({info['type']}) {info['model']}")
        return

    # Load config from YAML
    config_path = Path(args.config)
    if config_path.exists():
        cfg = load_experiment_config(config_path)
    else:
        print(f"Config not found: {config_path}, using defaults")
        cfg = ExperimentConfig()

    # Apply CLI overrides
    if args.llm is not None:
        cfg.llm = args.llm
    if args.ad_model is not None:
        cfg.ad_model = None if args.ad_model.lower() == "null" else args.ad_model
    if args.few_shot is not None:
        cfg.few_shot = args.few_shot
    if args.max_images is not None:
        cfg.max_images = args.max_images
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.resume is not None:
        cfg.resume = args.resume

    run_experiment(cfg)


if __name__ == "__main__":
    main()
