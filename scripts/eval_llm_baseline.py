"""
MMAD LLM Baseline Evaluation Script

Reproduces the paper's evaluation protocol:
- Few-shot normal templates + query image → LLM → MCQ answers
- Calculates accuracy per question type and dataset

Supports two modes:
1. Paper baseline (without AD model): Image + prompt → LLM
2. With AD model: AD model output (heatmap/bbox) + Image + prompt → LLM

Usage:
    # === API Models (GPT-4o, Claude) ===
    python scripts/eval_llm_baseline.py --model gpt-4o --few-shot 1 --similar-template --max-images 5
    python scripts/eval_llm_baseline.py --model claude --few-shot 1 --similar-template --max-images 5

    # === HuggingFace Models (Local GPU) ===
    # Qwen2.5-VL
    python scripts/eval_llm_baseline.py --model qwen --few-shot 1 --similar-template --max-images 5

    # InternVL2
    python scripts/eval_llm_baseline.py --model internvl --few-shot 1 --similar-template --max-images 5

    # LLaVA
    python scripts/eval_llm_baseline.py --model llava --few-shot 1 --similar-template --max-images 5

    # === With Anomaly Detection Model ===
    python scripts/eval_llm_baseline.py --model gpt-4o --with-ad --ad-output outputs/predictions.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

# Path setup
SCRIPT_PATH = Path(__file__).resolve()
PROJ_ROOT = SCRIPT_PATH.parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.mllm.factory import MODEL_REGISTRY, get_llm_client
from src.eval.metrics import calculate_accuracy_mmad


def load_mmad_data(json_path: str) -> dict:
    """Load MMAD dataset JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ad_predictions(ad_output_path: str) -> dict:
    """Load anomaly detection model predictions."""
    with open(ad_output_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    # Index by image path
    if isinstance(predictions, list):
        return {p["image_path"]: p for p in predictions}
    return predictions


def main():
    parser = argparse.ArgumentParser(description="MMAD LLM Baseline Evaluation")

    # Model settings
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="LLM model name (see MODEL_REGISTRY) or HuggingFace model path")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Override HuggingFace model path")

    # Data settings
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root path containing MMAD images")
    parser.add_argument("--mmad-json", type=str, default=None,
                        help="Path to mmad.json")

    # Evaluation settings
    parser.add_argument("--few-shot", type=int, default=1,
                        help="Number of few-shot examples (0-8)")
    parser.add_argument("--similar-template", action="store_true",
                        help="Use similar templates instead of random")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Max images to evaluate (for testing)")
    parser.add_argument("--batch-mode", action="store_true",
                        help="Ask all questions in one API call (faster but may be less accurate)")

    # Anomaly Detection model settings
    parser.add_argument("--with-ad", action="store_true",
                        help="Use anomaly detection model output")
    parser.add_argument("--ad-output", type=str, default=None,
                        help="Path to AD model predictions JSON")
    parser.add_argument("--ad-notation", type=str, default="heatmap",
                        choices=["bbox", "contour", "highlight", "mask", "heatmap"],
                        help="How to present AD model output to LLM")

    # Output settings
    parser.add_argument("--output-dir", type=str, default="outputs/eval",
                        help="Output directory for results")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results file")

    args = parser.parse_args()

    # Resolve paths
    data_root = args.data_root or os.environ.get("MMAD_DATA_ROOT")
    mmad_json = args.mmad_json or os.environ.get("MMAD_JSON_PATH")

    # Try common paths
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

    # Validate paths
    if not mmad_json or not Path(mmad_json).exists():
        print(f"Error: mmad.json not found at {mmad_json}")
        print("Please set --mmad-json or MMAD_JSON_PATH environment variable")
        sys.exit(1)

    if not data_root or not Path(data_root).exists():
        print(f"Error: Data root not found at {data_root}")
        print("Please set --data-root or MMAD_DATA_ROOT environment variable")
        sys.exit(1)

    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    template_type = "Similar_template" if args.similar_template else "Random_template"
    ad_suffix = "_with_AD" if args.with_ad else ""
    # Sanitize model name for filename (replace / with _)
    model_name_safe = args.model.replace("/", "_").replace("\\", "_")
    output_name = f"answers_{args.few_shot}_shot_{model_name_safe}_{template_type}{ad_suffix}"
    answers_json_path = output_dir / f"{output_name}.json"

    print("=" * 60)
    print("MMAD LLM Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    if args.model_path:
        print(f"Model path: {args.model_path}")
    print(f"Few-shot: {args.few_shot}")
    print(f"Template: {template_type}")
    print(f"With AD: {args.with_ad}")
    print(f"Data root: {data_root}")
    print(f"MMAD JSON: {mmad_json}")
    print(f"Output: {answers_json_path}")
    print("=" * 60)
    print()

    # Load AD predictions if using AD model
    ad_predictions = None
    if args.with_ad:
        if not args.ad_output:
            print("Error: --ad-output required when using --with-ad")
            sys.exit(1)
        ad_predictions = load_ad_predictions(args.ad_output)
        print(f"Loaded {len(ad_predictions)} AD predictions")

    # Load existing results if resuming
    all_answers = []
    existing_images = set()

    if args.resume and answers_json_path.exists():
        with open(answers_json_path, "r", encoding="utf-8") as f:
            all_answers = json.load(f)
        existing_images = {a["image"] for a in all_answers}
        print(f"Resuming from {len(existing_images)} existing images")

    # Initialize LLM client
    try:
        llm_client = get_llm_client(args.model, model_path=args.model_path)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Load dataset
    mmad_data = load_mmad_data(mmad_json)
    image_paths = list(mmad_data.keys())

    if args.max_images:
        image_paths = image_paths[:args.max_images]

    print(f"Total images: {len(image_paths)}")
    print()

    # Track statistics
    total_correct = 0
    total_questions = 0
    processed = 0
    errors = 0

    # Evaluate with progress bar
    pbar = tqdm(image_paths, desc="Evaluating", ncols=100)
    for image_rel in pbar:
        if image_rel in existing_images:
            continue

        meta = mmad_data[image_rel]

        # Get templates
        if args.similar_template:
            few_shot = meta.get("similar_templates", [])[:args.few_shot]
        else:
            few_shot = meta.get("random_templates", [])[:args.few_shot]

        # Build absolute paths
        query_image_path = str(Path(data_root) / image_rel)
        few_shot_paths = [str(Path(data_root) / p) for p in few_shot]

        # Check if image exists
        if not Path(query_image_path).exists():
            errors += 1
            continue

        # Generate answers
        if args.batch_mode:
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
                "gpt_answer": pred
            })

        # Save incrementally
        with open(answers_json_path, "w", encoding="utf-8") as f:
            json.dump(all_answers, f, indent=4, ensure_ascii=False)

    pbar.close()

    print()
    print("=" * 60)
    print("Calculating Metrics")
    print("=" * 60)
    calculate_accuracy_mmad(str(answers_json_path))
    print()
    print(f"Results saved to: {answers_json_path}")


if __name__ == "__main__":
    main()
