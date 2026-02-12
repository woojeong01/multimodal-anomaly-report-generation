"""Single-image test: InternVL report generation → PostgreSQL storage."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# allow `python scripts/test_report_pipeline.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mllm.factory import get_llm_client
from src.storage.pg import connect, get_report


def main():
    parser = argparse.ArgumentParser(description="Generate LLM report and save to PostgreSQL")
    parser.add_argument("--image", type=str, required=True, help="Path to product image")
    parser.add_argument("--category", type=str, default="unknown", help="Product category")
    parser.add_argument("--dataset", type=str, default="test", help="Dataset name")
    parser.add_argument("--model", type=str, default="internvl", help="LLM model name")
    parser.add_argument("--model-path", type=str, default=None, help="HuggingFace model path override")
    parser.add_argument("--dsn", type=str, default="postgresql://son:1234@localhost/inspection",
                        help="PostgreSQL connection string")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens for generation")
    args = parser.parse_args()

    image_path = str(Path(args.image).resolve())
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # 1. LLM client
    print(f"Loading LLM: {args.model} ...")
    client = get_llm_client(args.model, model_path=args.model_path, max_new_tokens=args.max_new_tokens)

    # 2. Generate report
    print(f"Generating report for: {image_path}")
    result = client.generate_report(
        image_path=image_path,
        category=args.category,
    )

    print(f"\n--- LLM Result ---")
    print(f"is_anomaly_LLM: {result['is_anomaly_LLM']}")
    print(f"llm_report: {json.dumps(result.get('llm_report'), ensure_ascii=False, indent=2)}")
    print(f"llm_summary: {json.dumps(result.get('llm_summary'), ensure_ascii=False, indent=2)}")
    print(f"inference_time: {result['llm_inference_duration']}s")

    # 3. Save to PostgreSQL
    print(f"\nSaving to PostgreSQL ...")
    conn = connect(args.dsn)

    from src.storage.pg import insert_report
    row = {
        "dataset": args.dataset,
        "category": args.category,
        "image_path": image_path,
        "is_anomaly_LLM": result["is_anomaly_LLM"],
        "llm_report": result["llm_report"],
        "llm_summary": result["llm_summary"],
        "llm_start_time": datetime.now(timezone.utc).isoformat(),
        "llm_inference_duration": result["llm_inference_duration"],
    }
    report_id = insert_report(conn, row)
    print(f"Saved! report id = {report_id}")

    # 4. Verify
    saved = get_report(conn, report_id)
    print(f"\n--- DB Record (id={report_id}) ---")
    print(json.dumps(saved, ensure_ascii=False, indent=2, default=str))

    conn.close()


if __name__ == "__main__":
    main()
