from __future__ import annotations

import argparse
from pathlib import Path
import sys
import csv

SCRIPT_PATH = Path(__file__).resolve()
PROJ_ROOT = SCRIPT_PATH.parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.datasets.mmad_index_csv import load_mmad_index_csv, filter_by_category, split_good_train_test
from src.datasets.anomalib_folder_builder import build_anomalib_folder_dataset
from src.utils.log import setup_logger

logger = setup_logger(name="PredictAnomalib", log_prefix="predict_anomalib")


def _import_model(model_name: str):
    try:
        from anomalib.models import Patchcore  # type: ignore
    except Exception:
        Patchcore = None

    EfficientAD = None
    for cand in ("EfficientAd", "EfficientAD", "Efficientad"):
        try:
            EfficientAD = getattr(__import__("anomalib.models", fromlist=[cand]), cand)
            break
        except Exception:
            continue

    WinCLIP = None
    for cand in ("WinCLIP", "WinClip"):
        try:
            WinCLIP = getattr(__import__("anomalib.models", fromlist=[cand]), cand)
            break
        except Exception:
            continue

    name = model_name.lower()
    if name == "patchcore":
        if Patchcore is None:
            raise ImportError("Could not import Patchcore from anomalib.models.")
        return Patchcore, dict(backbone="wide_resnet50_2", layers=["layer2", "layer3"], pre_trained=True)
    if name == "efficientad":
        if EfficientAD is None:
            raise ImportError("Could not import EfficientAD/EfficientAd from anomalib.models.")
        return EfficientAD, {}
    if name == "winclip":
        if WinCLIP is None:
            raise ImportError("Could not import WinCLIP/WinClip from anomalib.models.")
        return WinCLIP, {}
    raise ValueError(f"Unknown model: {model_name}")


def _extract_score(pred) -> float:
    # pred can be dict, object with attributes, or tensor
    for k in ("pred_score", "pred_scores", "score", "anomaly_score"):
        if isinstance(pred, dict) and k in pred:
            v = pred[k]
            try:
                return float(v.item())  # torch tensor
            except Exception:
                try:
                    return float(v)
                except Exception:
                    pass
        if hasattr(pred, k):
            v = getattr(pred, k)
            try:
                return float(v.item())
            except Exception:
                try:
                    return float(v)
                except Exception:
                    pass
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict anomalib models on MMAD_index.csv")
    ap.add_argument("--index-csv", type=str, required=True)
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument(
        "--category",
        type=str,
        required=True,
        help="Category name (e.g., bottle). Use 'all' to predict each category separately.",
    )
    ap.add_argument("--model", type=str, default="patchcore", choices=["patchcore", "efficientad", "winclip"])
    ap.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to checkpoint (.ckpt). If omitted, script will auto-find under --trained-root/<model>/<category>/.",
    )
    ap.add_argument(
        "--trained-root",
        type=str,
        default="outputs_anomalib",
        help="Where train_anomalib.py wrote checkpoints (used for auto-find).",
    )
    ap.add_argument("--work-dir", type=str, default="data_anomalib")
    ap.add_argument("--out-dir", type=str, default="predictions_anomalib")
    ap.add_argument("--image-size", type=int, nargs=2, default=[1024, 1024])
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy-files", action="store_true")
    args = ap.parse_args()

    def find_ckpt_for_category(category: str) -> Path:
        if args.ckpt:
            ckpt = Path(args.ckpt)
            if not ckpt.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
            return ckpt

        base = Path(args.trained_root) / args.model / category
        if not base.exists():
            raise FileNotFoundError(
                f"Could not auto-find checkpoint. Folder not found: {base}. Provide --ckpt or set --trained-root correctly."
            )
        # common anomalib path
        cand = base / "weights" / "lightning" / "model.ckpt"
        if cand.exists():
            return cand

        ckpts = list(base.rglob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt found under: {base}")
        ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return ckpts[0]

    def run_one_category(category: str) -> Path:
        ckpt = find_ckpt_for_category(category)

        records = load_mmad_index_csv(args.index_csv, data_root=args.data_root)
        cat_records = filter_by_category(records, category)
        if not cat_records:
            raise ValueError(f"No records for category='{category}'.")
        train_goods, test_records = split_good_train_test(cat_records, train_ratio=args.train_ratio, seed=args.seed)

        built = build_anomalib_folder_dataset(
            train_goods=train_goods,
            test_records=test_records,
            out_root=Path(args.work_dir),
            category=category,
            copy_files=bool(args.copy_files),
        )
        cat_root = Path(built.root) / built.category

        from anomalib.data import Folder  # type: ignore
        from anomalib.engine import Engine  # type: ignore

        # anomalib's Folder datamodule signature changes across versions.
        # Some releases accept `image_size`, others don't.
        import inspect

        def _safe_folder_init(**kwargs):
            sig = inspect.signature(Folder.__init__)
            allowed = set(sig.parameters.keys())
            filtered = {k: v for k, v in kwargs.items() if k in allowed}
            return Folder(**filtered)

        ModelCls, model_kwargs = _import_model(args.model)
        model = ModelCls(**model_kwargs) if model_kwargs else ModelCls()

        datamodule = _safe_folder_init(
            name=category,
            root=str(cat_root),
            normal_dir="train/good",
            image_size=tuple(args.image_size),
            train_batch_size=1,  # unused in predict
            eval_batch_size=int(args.eval_batch_size),
            num_workers=int(args.num_workers),
        )
        datamodule.setup()

        out_dir = Path(args.out_dir) / args.model / category
        out_dir.mkdir(parents=True, exist_ok=True)

        engine = Engine(task="segmentation", default_root_dir=str(out_dir), max_epochs=1)
        preds = engine.predict(model=model, datamodule=datamodule, ckpt_path=str(ckpt))

        # Flatten predictions
        rows = []
        if preds is None:
            preds = []
        for batch in preds:
            if isinstance(batch, list):
                rows.extend(batch)
            else:
                rows.append(batch)

        out_csv = out_dir / "pred_scores.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index", "score"])
            for i, p in enumerate(rows):
                w.writerow([i, _extract_score(p)])

        logger.info(f"Saved prediction scores: {out_csv}")
        return out_csv

    # category=all
    all_records = load_mmad_index_csv(args.index_csv, data_root=args.data_root)
    categories = sorted({r.category for r in all_records})
    if args.category.lower() == "all":
        outs: list[Path] = []
        for cat in categories:
            logger.info(f"=== predict [{args.model}] category: {cat} ===")
            outs.append(run_one_category(cat))
        print("\n".join(str(p) for p in outs))
        return

    out_csv = run_one_category(args.category)
    print(str(out_csv))


if __name__ == "__main__":
    main()
