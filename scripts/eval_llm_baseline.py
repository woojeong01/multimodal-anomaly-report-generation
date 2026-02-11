"""
MMAD LLM Baseline Evaluation Script

Reproduces the paper's evaluation protocol:
- Few-shot normal templates + query image → LLM → MCQ answers
- Calculates accuracy per question type and dataset

Supports two modes:
1. Paper baseline (without AD model): Image + prompt → LLM
2. With AD model: AD model output (heatmap/bbox) + Image + prompt → LLM

Performance Optimizations (v2):
- API models: Batch mode by default (1 API call per image, 5-8x faster)
- Local models: Incremental mode by default (better accuracy)
- Parallel workers: Multiple concurrent API calls (API models only)
- Buffered I/O: Save every 50 images instead of every image

Usage:
    # === API Models (auto batch mode, fast) ===
    python scripts/eval_llm_baseline.py --model gpt-4o --few-shot 1 --similar-template
    python scripts/eval_llm_baseline.py --model claude --few-shot 1 --similar-template
    python scripts/eval_llm_baseline.py --model gemini --few-shot 1 --similar-template

    # === With parallel workers (API models, 2-3x faster) ===
    python scripts/eval_llm_baseline.py --model gpt-4o --parallel 3

    # === Local Models (auto incremental mode, accurate) ===
    python scripts/eval_llm_baseline.py --model qwen --few-shot 1 --similar-template
    python scripts/eval_llm_baseline.py --model internvl --few-shot 1 --similar-template
    python scripts/eval_llm_baseline.py --model llava --few-shot 1 --similar-template

    # === Force batch mode for local models (faster but less accurate) ===
    python scripts/eval_llm_baseline.py --model llava --batch-mode

    # === Quick experiments with sampling (10% stratified sample) ===
    python scripts/eval_llm_baseline.py --model llava --sample-ratio 0.1
    python scripts/eval_llm_baseline.py --model llava --sample-ratio 0.2 --sample-seed 123

    # === With Anomaly Detection Model ===
    python scripts/eval_llm_baseline.py --model gpt-4o --with-ad --ad-output output/ad_predictions.json

    # === Quick test ===
    python scripts/eval_llm_baseline.py --model gpt-4o --max-images 10
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
    """Load anomaly detection model predictions.

    Supports multiple JSON formats:
    1. List format: [{"image_path": "...", ...}, ...]
    2. Dict format with image paths as keys: {"path/to/image.jpg": {...}, ...}
    3. Dict format with "predictions" key: {"predictions": [...]}

    Returns:
        Dictionary indexed by image relative path (e.g., "GoodsAD/cigarette_box/test/bad/001.jpg")
    """
    with open(ad_output_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    # Handle different formats
    if isinstance(predictions, list):
        # List format - index by image_path
        indexed = {}
        for p in predictions:
            # Try common key names for image path
            img_key = p.get("image_path") or p.get("image") or p.get("img_path") or p.get("path")
            if img_key:
                # Normalize path (remove leading ./ or data_root prefix if present)
                img_key = img_key.lstrip("./")
                indexed[img_key] = p
        return indexed
    elif isinstance(predictions, dict):
        # Check if it has a "predictions" key
        if "predictions" in predictions:
            return load_ad_predictions_from_list(predictions["predictions"])
        # Already indexed by image path
        return predictions
    return predictions


def load_ad_predictions_from_list(predictions_list: list) -> dict:
    """Helper to index predictions list by image path."""
    indexed = {}
    for p in predictions_list:
        img_key = p.get("image_path") or p.get("image") or p.get("img_path") or p.get("path")
        if img_key:
            img_key = img_key.lstrip("./")
            indexed[img_key] = p
    return indexed


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
    parser.add_argument("--sample-ratio", type=float, default=None,
                        help="Sample ratio (0.0-1.0) for quick experiments. E.g., 0.1 for 10%%. Uses stratified sampling by category.")
    parser.add_argument("--sample-seed", type=int, default=42,
                        help="Random seed for sampling (default: 42, for reproducibility)")
    parser.add_argument("--batch-mode", action="store_true",
                        help="Ask all questions in one API call (faster, good for API models)")
    parser.add_argument("--incremental-mode", action="store_true",
                        help="Ask questions one by one (slower but more accurate, default for local models)")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="Save results every N images (default: 50)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel workers for API calls (default: 1, max: 5)")

    # Anomaly Detection model settings
    parser.add_argument("--with-ad", action="store_true",
                        help="Use anomaly detection model output")
    parser.add_argument("--ad-output", type=str, default=None,
                        help="Path to AD model predictions JSON")
    parser.add_argument("--ad-notation", type=str, default="heatmap",
                        choices=["bbox", "contour", "highlight", "mask", "heatmap"],
                        help="How to present AD model output to LLM")

    # Output settings
    parser.add_argument("--output-dir", type=str, default="output/eval",
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
    # Determine batch mode
    # - API models (GPT-4o, Claude, Gemini): batch mode by default (5-8x faster)
    # - Local models (LLaVA, Qwen, InternVL): incremental mode by default (better accuracy)
    # User can override with --batch-mode or --incremental-mode
    model_info = MODEL_REGISTRY.get(args.model.lower(), {})
    is_local_model = model_info.get("type") == "local"

    if args.batch_mode and args.incremental_mode:
        print("Warning: Both --batch-mode and --incremental-mode specified. Using incremental.")
        use_batch_mode = False
    elif args.batch_mode:
        use_batch_mode = True
    elif args.incremental_mode:
        use_batch_mode = False
    else:
        # Auto-detect based on model type
        use_batch_mode = not is_local_model  # API models use batch, local models use incremental

    print(f"Model: {args.model}")
    if args.model_path:
        print(f"Model path: {args.model_path}")
    print(f"Few-shot: {args.few_shot}")
    print(f"Template: {template_type}")
    print(f"Mode: {'Batch (1 API call/image)' if use_batch_mode else 'Incremental (N API calls/image)'}")
    print(f"Parallel workers: {args.parallel}")
    print(f"Save interval: {args.save_interval} images")
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
    total_available = len(image_paths)

    # Stratified sampling by category (for quick experiments)
    if args.sample_ratio is not None:
        import random
        from collections import defaultdict

        random.seed(args.sample_seed)
        ratio = max(0.0, min(1.0, args.sample_ratio))

        # Group images by category (dataset/class)
        images_by_category = defaultdict(list)
        for img_path in image_paths:
            parts = img_path.split("/")
            if len(parts) >= 2:
                category = f"{parts[0]}/{parts[1]}"
            else:
                category = "unknown"
            images_by_category[category].append(img_path)

        # Sample from each category proportionally
        sampled_paths = []
        for category, paths in images_by_category.items():
            n_sample = max(1, int(len(paths) * ratio))  # At least 1 per category
            sampled = random.sample(paths, min(n_sample, len(paths)))
            sampled_paths.extend(sampled)

        random.shuffle(sampled_paths)  # Shuffle to mix categories
        image_paths = sampled_paths
        print(f"Stratified sampling: {ratio*100:.0f}% from {len(images_by_category)} categories")
        print(f"  Total: {total_available} -> Sampled: {len(image_paths)}")

    if args.max_images:
        image_paths = image_paths[:args.max_images]

    print(f"Images to evaluate: {len(image_paths)}")
    print()

    # Track statistics
    total_correct = 0
    total_questions = 0
    processed = 0
    errors = 0
    last_save = 0

    def process_single_image(image_rel):
        """Process a single image and return results."""
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
            return None, "not_found"

        # Get AD prediction for this image if available
        ad_info = None
        if ad_predictions is not None:
            ad_info = ad_predictions.get(image_rel)
            if ad_info is None:
                normalized_path = image_rel.replace("\\", "/")
                ad_info = ad_predictions.get(normalized_path)

        # Generate answers
        if use_batch_mode:
            questions, answers, predicted, q_types = llm_client.generate_answers_batch(
                query_image_path, meta, few_shot_paths, ad_info=ad_info
            )
        else:
            questions, answers, predicted, q_types = llm_client.generate_answers(
                query_image_path, meta, few_shot_paths, ad_info=ad_info
            )

        if predicted is None or len(predicted) != len(answers):
            return None, "failed"

        return {
            "image_rel": image_rel,
            "questions": questions,
            "answers": answers,
            "predicted": predicted,
            "q_types": q_types
        }, "success"

    # Filter images to process
    images_to_process = [img for img in image_paths if img not in existing_images]
    print(f"Images to process: {len(images_to_process)} (skipping {len(existing_images)} existing)")

    # Parallel or sequential processing
    if args.parallel > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        num_workers = min(args.parallel, 5)  # Cap at 5 workers
        print(f"Using {num_workers} parallel workers")

        pbar = tqdm(total=len(images_to_process), desc="Evaluating", ncols=100)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_image, img): img for img in images_to_process}

            for future in as_completed(futures):
                result, status = future.result()

                if status == "success" and result:
                    correct = sum(1 for p, a in zip(result["predicted"], result["answers"]) if p == a)
                    total_correct += correct
                    total_questions += len(result["answers"])
                    processed += 1

                    for q, a, pred, qt in zip(result["questions"], result["answers"], result["predicted"], result["q_types"]):
                        all_answers.append({
                            "image": result["image_rel"],
                            "question": q,
                            "question_type": qt,
                            "correct_answer": a,
                            "gpt_answer": pred
                        })
                else:
                    errors += 1

                # Update progress bar
                running_acc = total_correct / total_questions if total_questions > 0 else 0
                pbar.update(1)
                pbar.set_postfix({"acc": f"{running_acc:.1%}", "done": processed, "err": errors}, refresh=False)

                # Save periodically
                if processed - last_save >= args.save_interval:
                    with open(answers_json_path, "w", encoding="utf-8") as f:
                        json.dump(all_answers, f, indent=2, ensure_ascii=False)
                    last_save = processed

        pbar.close()
    else:
        # Sequential processing
        pbar = tqdm(images_to_process, desc="Evaluating", ncols=100)
        for image_rel in pbar:
            result, status = process_single_image(image_rel)

            if status == "success" and result:
                correct = sum(1 for p, a in zip(result["predicted"], result["answers"]) if p == a)
                total_correct += correct
                total_questions += len(result["answers"])
                processed += 1

                for q, a, pred, qt in zip(result["questions"], result["answers"], result["predicted"], result["q_types"]):
                    all_answers.append({
                        "image": result["image_rel"],
                        "question": q,
                        "question_type": qt,
                        "correct_answer": a,
                        "gpt_answer": pred
                    })
            else:
                errors += 1

            # Update progress bar
            running_acc = total_correct / total_questions if total_questions > 0 else 0
            pbar.set_postfix({"acc": f"{running_acc:.1%}", "done": processed, "err": errors}, refresh=False)

            # Save periodically (not every image!)
            if processed - last_save >= args.save_interval:
                with open(answers_json_path, "w", encoding="utf-8") as f:
                    json.dump(all_answers, f, indent=2, ensure_ascii=False)
                last_save = processed

        pbar.close()

    # Final save
    with open(answers_json_path, "w", encoding="utf-8") as f:
        json.dump(all_answers, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("Calculating Metrics")
    print("=" * 60)
    calculate_accuracy_mmad(str(answers_json_path))
    print()
    print(f"Results saved to: {answers_json_path}")


if __name__ == "__main__":
    main()
