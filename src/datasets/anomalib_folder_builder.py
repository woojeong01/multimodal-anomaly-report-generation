from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .mmad_index_csv import MMADIndexRecord


def _safe_link(src: Path, dst: Path, *, copy: bool = False) -> None:
    """Create a symlink (preferred) or copy if requested / unsupported.

    On Windows, symlink may require admin privileges. In that case, we fall back to copy.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


@dataclass(frozen=True)
class BuiltFolderDataset:
    root: Path
    category: str


def build_anomalib_folder_dataset(
    train_goods: List[MMADIndexRecord],
    test_records: List[MMADIndexRecord],
    *,
    out_root: str | Path,
    category: str,
    copy_files: bool = False,
) -> BuiltFolderDataset:
    """Build an anomalib Folder-format dataset from MMAD_index.csv records.

    Result structure:
      <out_root>/<category>/
        train/good/*.png
        test/good/*.png
        test/<defect_type>/*.png
        ground_truth/<defect_type>/*.png   (optional, if masks available)

    Notes
    -----
    - PatchCore/EfficientAD training uses train/good.
    - Segmentation evaluation uses ground_truth/<defect_type>.
    """
    out_root = Path(out_root)
    cat_root = out_root / category

    # clean only category folder (safe)
    if cat_root.exists():
        shutil.rmtree(cat_root)
    (cat_root / "train" / "good").mkdir(parents=True, exist_ok=True)
    (cat_root / "test" / "good").mkdir(parents=True, exist_ok=True)

    # ---- train goods ----
    for i, r in enumerate(train_goods):
        if not r.image_path.exists():
            raise FileNotFoundError(f"Missing image: {r.image_path}")
        # anomalib's folder dataset extension filter can be case-sensitive in some versions.
        # Normalize to lowercase to avoid "Found 0 ... images" when files are like ".PNG".
        ext = (r.image_path.suffix or ".png").lower()
        dst = cat_root / "train" / "good" / f"{i:06d}{ext}"
        _safe_link(r.image_path, dst, copy=copy_files)

    # ---- test records (goods + bads) ----
    good_i = 0
    bad_counts: Dict[str, int] = {}
    for r in test_records:
        if not r.image_path.exists():
            raise FileNotFoundError(f"Missing image: {r.image_path}")

        if r.is_good:
            ext = (r.image_path.suffix or ".png").lower()
            dst = cat_root / "test" / "good" / f"{good_i:06d}{ext}"
            good_i += 1
            _safe_link(r.image_path, dst, copy=copy_files)
            continue

        defect = r.defect_type
        bad_counts.setdefault(defect, 0)
        bad_counts[defect] += 1

        ext = (r.image_path.suffix or ".png").lower()
        dst = cat_root / "test" / defect / f"{bad_counts[defect]-1:06d}{ext}"
        _safe_link(r.image_path, dst, copy=copy_files)

        if r.mask_path is not None and r.mask_path.exists():
            mext = (r.mask_path.suffix or ".png").lower()
            mdst = cat_root / "ground_truth" / defect / f"{bad_counts[defect]-1:06d}{mext}"
            _safe_link(r.mask_path, mdst, copy=copy_files)

    return BuiltFolderDataset(root=out_root, category=category)
