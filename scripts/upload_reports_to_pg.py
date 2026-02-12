"""Read JSON report file(s) and insert into PostgreSQL."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.storage.pg import connect, insert_report


def main():
    parser = argparse.ArgumentParser(description="Upload JSON reports to PostgreSQL")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to a .json file or directory of .json files")
    parser.add_argument("--dsn", type=str, default="postgresql://son:1234@localhost/inspection",
                        help="PostgreSQL connection string")
    args = parser.parse_args()

    conn = connect(args.dsn)
    input_path = Path(args.input)

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.glob("*.json"))
    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)

    if not files:
        print("No JSON files found.")
        sys.exit(1)

    print(f"Found {len(files)} file(s). Uploading...")

    count = 0
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        # 단건 dict or 리스트 모두 지원
        records = data if isinstance(data, list) else [data]

        for record in records:
            report_id = insert_report(conn, record)
            count += 1
            print(f"  [{count}] id={report_id} | {record.get('category', '')} | {record.get('image_path', '')}")

    conn.close()
    print(f"\nDone. {count} report(s) inserted.")


if __name__ == "__main__":
    main()
