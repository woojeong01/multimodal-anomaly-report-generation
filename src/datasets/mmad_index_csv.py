from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd


LIST_COLS = ("similar_templates", "random_templates")


@dataclass(frozen=True)
class MMADIndexRecord:
    """A single record parsed from MMAD_index.csv.

    All paths are stored as POSIX-like relative strings in the CSV. We resolve them to
    absolute Paths using a `data_root` provided by the user.
    """
    key: str
    image_path: Path
    mask_path: Optional[Path]
    source: str
    category: str
    good_bad: str
    similar_templates: List[str]
    random_templates: List[str]

    @property
    def is_good(self) -> bool:
        return self.good_bad.strip().lower() == "good"

    @property
    def defect_type(self) -> str:
        """Defect type name inferred from image_path.

        Example: DS-MVTec/bottle/image/broken_large/000.png -> broken_large
        If not inferable, returns 'bad'.
        """
        parts = self.image_path.as_posix().split("/")
        # look for .../image/<defect_type>/<file>
        if "image" in parts:
            i = parts.index("image")
            if i + 1 < len(parts) - 1:
                return parts[i + 1]
        return "bad"


def _parse_list_cell(v: Any) -> List[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):  # NaN
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    s = str(v).strip()
    if s == "":
        return []
    # CSV stores Python-like list strings, e.g. "['a.png','b.png']"
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    # fallback: single item
    return [s]


def load_mmad_index_csv(csv_path: str | Path, *, data_root: str | Path) -> List[MMADIndexRecord]:
    """Load MMAD_index.csv and resolve relative paths using `data_root`.

    Parameters
    ----------
    csv_path: path to MMAD_index.csv
    data_root: directory that contains the relative paths referenced by the CSV.
              e.g. if CSV has 'DS-MVTec/..', then data_root/'DS-MVTec/..' must exist.
    """
    csv_path = Path(csv_path)
    data_root = Path(data_root)

    df = pd.read_csv(csv_path)

    required = {"key", "image_path", "source", "category", "good_bad", "mask_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"MMAD_index.csv missing required columns: {sorted(missing)}")

    records: List[MMADIndexRecord] = []
    for _, row in df.iterrows():
        img_rel = Path(str(row["image_path"]))
        mask_rel = str(row.get("mask_path", "")).strip()
        mask_p = Path(mask_rel) if mask_rel and mask_rel.lower() != "nan" else None

        records.append(
            MMADIndexRecord(
                key=str(row["key"]),
                image_path=(data_root / img_rel).resolve(),
                mask_path=(data_root / mask_p).resolve() if mask_p else None,
                source=str(row["source"]),
                category=str(row["category"]),
                good_bad=str(row["good_bad"]),
                similar_templates=_parse_list_cell(row.get("similar_templates")),
                random_templates=_parse_list_cell(row.get("random_templates")),
            )
        )
    return records


def filter_by_category(records: List[MMADIndexRecord], category: str) -> List[MMADIndexRecord]:
    cat = category.strip()
    return [r for r in records if r.category == cat]


def split_good_train_test(
    records: List[MMADIndexRecord],
    *,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[List[MMADIndexRecord], List[MMADIndexRecord]]:
    """Split *good* samples into train/test while keeping all bad samples in test.

    This is needed because MMAD_index.csv does not store an explicit train/test split.
    - Train: subset of good samples
    - Test: remaining good samples + all bad samples
    """
    import random

    goods = [r for r in records if r.is_good]
    bads = [r for r in records if not r.is_good]

    # Some categories in MMAD_index.csv may contain only anomalous samples.
    # In that case, return an empty train split and keep everything in test.
    # The caller can decide to skip such categories.
    if not goods:
        return [], bads

    rng = random.Random(seed)
    idx = list(range(len(goods)))
    rng.shuffle(idx)

    n_train = max(1, int(len(goods) * float(train_ratio)))
    train_goods = [goods[i] for i in idx[:n_train]]
    test_goods = [goods[i] for i in idx[n_train:]]

    test = test_goods + bads
    return train_goods, test
