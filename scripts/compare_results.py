"""Compare experiment results from .meta.json files.

Scans outputs/eval/ for metadata files and prints a comparison table.

Usage:
    python scripts/compare_results.py
    python scripts/compare_results.py --output-dir outputs/eval
    python scripts/compare_results.py --sort accuracy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJ_ROOT = SCRIPT_PATH.parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))


def collect_results(output_dir: str) -> list[dict]:
    """Scan directory for .meta.json files and return list of result dicts."""
    results = []
    output_path = Path(output_dir)

    if not output_path.exists():
        return results

    for meta_file in sorted(output_path.glob("*.meta.json")):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["_file"] = str(meta_file)
            results.append(meta)
        except (json.JSONDecodeError, KeyError):
            continue

    return results


def print_table(results: list[dict], sort_by: str = "timestamp"):
    """Print results as a formatted comparison table."""
    if not results:
        print("No experiment results found.")
        return

    # Sort
    if sort_by == "accuracy":
        results.sort(key=lambda r: r.get("accuracy", 0), reverse=True)
    elif sort_by == "name":
        results.sort(key=lambda r: r.get("experiment_name", ""))
    else:
        results.sort(key=lambda r: r.get("timestamp", ""))

    # Header
    headers = ["Experiment", "LLM", "AD Model", "Few-shot", "Accuracy", "Images", "Errors", "Time", "s/img", "Timestamp"]
    widths = [25, 15, 15, 8, 8, 8, 6, 8, 7, 20]

    header_line = ""
    for h, w in zip(headers, widths):
        header_line += f"{h:<{w}}"
    print("=" * sum(widths))
    print(header_line)
    print("-" * sum(widths))

    # Rows
    for r in results:
        elapsed = r.get("elapsed_seconds", 0)
        processed = r.get("processed") or 0
        sec_per_img = elapsed / processed if processed > 0 else 0
        row = [
            r.get("experiment_name", "?")[:24],
            r.get("llm", "?")[:14],
            (r.get("ad_model") or "none")[:14],
            str(r.get("few_shot", "?")),
            f"{r.get('accuracy', 0):.1f}%",
            str(processed),
            str(r.get("errors", 0)),
            f"{elapsed:.0f}s",
            f"{sec_per_img:.1f}s",
            r.get("timestamp", "?")[:19],
        ]
        line = ""
        for val, w in zip(row, widths):
            line += f"{val:<{w}}"
        print(line)

    print("=" * sum(widths))
    print(f"\nTotal experiments: {len(results)}")

    # Best result
    if any(r.get("accuracy", 0) > 0 for r in results):
        best = max(results, key=lambda r: r.get("accuracy", 0))
        print(f"Best accuracy: {best.get('accuracy', 0):.1f}% ({best.get('experiment_name', '?')})")


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("--output-dir", type=str, nargs="+", default=["outputs/eval"],
                        help="Directory(s) containing .meta.json files (space-separated)")
    parser.add_argument("--sort", type=str, default="timestamp",
                        choices=["timestamp", "accuracy", "name"],
                        help="Sort results by field")
    parser.add_argument("--filter-images", type=int, default=None,
                        help="Only show results with this many processed images")
    parser.add_argument("--filter-llm", type=str, default=None,
                        help="Only show results matching this LLM name (substring match)")
    parser.add_argument("--max-sec-per-img", type=float, default=None,
                        help="Only show results with s/img <= this value")
    args = parser.parse_args()

    results = []
    for d in args.output_dir:
        results.extend(collect_results(d))

    if args.filter_images is not None:
        results = [r for r in results if r.get("processed") == args.filter_images]

    if args.filter_llm is not None:
        results = [r for r in results if args.filter_llm in (r.get("llm") or "")]

    if args.max_sec_per_img is not None:
        def _sec_per_img(r):
            elapsed = r.get("elapsed_seconds", 0)
            processed = r.get("processed") or 0
            return elapsed / processed if processed > 0 else float("inf")
        results = [r for r in results if _sec_per_img(r) <= args.max_sec_per_img]

    print_table(results, sort_by=args.sort)


if __name__ == "__main__":
    main()
