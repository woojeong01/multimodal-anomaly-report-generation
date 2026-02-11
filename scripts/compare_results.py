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
    headers = ["Experiment", "LLM", "AD Model", "Few-shot", "Accuracy", "Images", "Errors", "Time", "Timestamp"]
    widths = [25, 15, 15, 8, 8, 8, 6, 8, 20]

    header_line = ""
    for h, w in zip(headers, widths):
        header_line += f"{h:<{w}}"
    print("=" * sum(widths))
    print(header_line)
    print("-" * sum(widths))

    # Rows
    for r in results:
        row = [
            r.get("experiment_name", "?")[:24],
            r.get("llm", "?")[:14],
            (r.get("ad_model") or "none")[:14],
            str(r.get("few_shot", "?")),
            f"{r.get('accuracy', 0):.1f}%",
            str(r.get("processed", "?")),
            str(r.get("errors", 0)),
            f"{r.get('elapsed_seconds', 0):.0f}s",
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
    parser.add_argument("--output-dir", type=str, default="outputs/eval",
                        help="Directory containing .meta.json files")
    parser.add_argument("--sort", type=str, default="timestamp",
                        choices=["timestamp", "accuracy", "name"],
                        help="Sort results by field")
    args = parser.parse_args()

    results = collect_results(args.output_dir)
    print_table(results, sort_by=args.sort)


if __name__ == "__main__":
    main()
