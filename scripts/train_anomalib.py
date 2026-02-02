from __future__ import annotations

import argparse
from pathlib import Path
import sys
import glob

# Make repo importable when executed as script
SCRIPT_PATH = Path(__file__).resolve()
PROJ_ROOT = SCRIPT_PATH.parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.datasets.mmad_index_csv import load_mmad_index_csv, filter_by_category, split_good_train_test
from src.datasets.anomalib_folder_builder import build_anomalib_folder_dataset
from src.utils.log import setup_logger

logger = setup_logger(name="TrainAnomalib", log_prefix="train_anomalib")


def _import_model(model_name: str):
    """Import anomalib model class in a version-tolerant way."""
    try:
        from anomalib.models import Patchcore  # type: ignore
    except Exception:
        Patchcore = None

    # EfficientAD naming varies across versions: EfficientAd / EfficientAD
    EfficientAD = None
    for cand in ("EfficientAd", "EfficientAD", "Efficientad"):
        try:
            EfficientAD = getattr(__import__("anomalib.models", fromlist=[cand]), cand)
            break
        except Exception:
            continue

    # WinCLIP naming varies: WinClip / WinCLIP
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
            raise ImportError("Could not import Patchcore from anomalib.models. Please check anomalib installation.")
        return Patchcore
    if name == "efficientad":
        if EfficientAD is None:
            raise ImportError("Could not import EfficientAD/EfficientAd from anomalib.models. Please check anomalib version.")
        return EfficientAD
    if name == "winclip":
        if WinCLIP is None:
            raise ImportError("Could not import WinCLIP/WinClip from anomalib.models. Please check anomalib version.")
        return WinCLIP

    raise ValueError(f"Unknown model: {model_name}. Choose from: patchcore, efficientad, winclip")


def _find_ckpt(output_dir: Path) -> Path:
    # anomalib(Engine/Lightning) usually writes: <out>/weights/lightning/model.ckpt
    candidates = [
        output_dir / "weights" / "lightning" / "model.ckpt",
    ]
    for c in candidates:
        if c.exists():
            return c

    # fallback: pick newest .ckpt
    ckpts = list(output_dir.rglob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found under: {output_dir}")
    ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train anomalib models using MMAD_index.csv")
    ap.add_argument("--index-csv", type=str, required=True, help="Path to MMAD_index.csv")
    ap.add_argument("--data-root", type=str, required=True, help="Dataset root containing relative paths from CSV")
    ap.add_argument(
        "--category",
        type=str,
        required=True,
        help="Category name (e.g., bottle). Use 'all' to train/fit each category separately.",
    )
    ap.add_argument("--model", type=str, default="patchcore", choices=["patchcore", "efficientad", "winclip"])
    ap.add_argument("--train-ratio", type=float, default=0.9, help="Fraction of good samples used for training")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--work-dir", type=str, default="data_anomalib", help="Folder dataset will be built here")
    ap.add_argument("--output-dir", type=str, default="outputs_anomalib", help="Training outputs directory")
    ap.add_argument("--image-size", type=int, nargs=2, default=[1024, 1024])
    ap.add_argument("--train-batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--max-epochs",
        type=int,
        default=1,
        help="Training epochs (only used for EfficientAD). PatchCore/WinCLIP are fit-only.",
    )
    ap.add_argument("--copy-files", action="store_true", help="Copy images instead of symlink (Windows-safe)")
    args = ap.parse_args()

    def effective_max_epochs(model_name: str, requested: int) -> int:
        """EfficientAD is trainable; PatchCore/WinCLIP are effectively fit-only."""
        if model_name.lower() == "efficientad":
            return int(requested)
        return 1

    def run_one_category(category: str) -> Path | None:
        # load + filter
        records = load_mmad_index_csv(args.index_csv, data_root=args.data_root)
        cat_records = filter_by_category(records, category)
        if not cat_records:
            raise ValueError(
                f"No records for category='{category}'. Available categories: {sorted({r.category for r in records})[:50]}"
            )

        # Split good samples into train/test. Some categories may have very few or no good samples
        # depending on the CSV contents. We'll handle empty-train safely below.
        train_goods, test_records = split_good_train_test(cat_records, train_ratio=args.train_ratio, seed=args.seed)

        # build folder dataset
        work_dir = Path(args.work_dir)
        built = build_anomalib_folder_dataset(
            train_goods=train_goods,
            test_records=test_records,
            out_root=work_dir,
            category=category,
            copy_files=bool(args.copy_files),
        )
        cat_root = Path(built.root) / built.category
        logger.info(f"Built Folder dataset at: {cat_root}")

        # Safety: anomalib Folder dataset will crash if train/good has zero images.
        # This can happen if a category has no usable good samples or extensions mismatch.
        train_good_dir = cat_root / "train" / "good"
        train_imgs = [p for p in train_good_dir.glob("*") if p.is_file()]
        if len(train_imgs) == 0:
            logger.warning(
                f"Skipping category='{category}' because no normal images were found in {train_good_dir}. "
                "(The CSV may contain no 'good' samples for this category, or paths may be filtered.)"
            )
            return None

        # import anomalib pieces
        from anomalib.data import Folder  # type: ignore
        from anomalib.engine import Engine  # type: ignore

        # anomalib's Folder datamodule signature changes across versions.
        # Example: some versions accept `image_size`, others don't.
        # To prevent hard failures, we filter kwargs by the runtime signature.
        import inspect

        def _safe_folder_init(**kwargs):
            sig = inspect.signature(Folder.__init__)
            allowed = set(sig.parameters.keys())
            filtered = {k: v for k, v in kwargs.items() if k in allowed}
            return Folder(**filtered)

        ModelCls = _import_model(args.model)

        model_kwargs = {}
        if args.model == "patchcore":
            model_kwargs = dict(backbone="wide_resnet50_2", layers=["layer2", "layer3"], pre_trained=True)

        model = ModelCls(**model_kwargs) if model_kwargs else ModelCls()

        datamodule = _safe_folder_init(
            name=category,
            root=str(cat_root),
            normal_dir="train/good",
            image_size=tuple(args.image_size),
            train_batch_size=int(args.train_batch_size),
            eval_batch_size=int(args.eval_batch_size),
            num_workers=int(args.num_workers),
        )
        datamodule.setup()

        out_dir = Path(args.output_dir) / args.model / category
        out_dir.mkdir(parents=True, exist_ok=True)

        max_epochs = effective_max_epochs(args.model, args.max_epochs)
        if args.model in ("patchcore", "winclip"):
            logger.info(f"Model '{args.model}' is fit-only. max_epochs is forced to {max_epochs}.")
        else:
            logger.info(f"Model '{args.model}' will be trained for max_epochs={max_epochs}.")

        engine = Engine(
            task="segmentation",
            default_root_dir=str(out_dir),
            max_epochs=max_epochs,
        )
        engine.fit(model=model, datamodule=datamodule)

        ckpt = _find_ckpt(out_dir)
        logger.info(f"Done. Checkpoint: {ckpt}")
        return ckpt

    # category=all: loop over all categories in CSV
    all_records = load_mmad_index_csv(args.index_csv, data_root=args.data_root)
    categories = sorted({r.category for r in all_records})

    if args.category.lower() == "all":
        ckpts: list[Path] = []
        for cat in categories:
            logger.info(f"=== [{args.model}] category: {cat} ===")
            ckpt = run_one_category(cat)
            if ckpt is not None:
                ckpts.append(ckpt)
        # print all ckpts for convenience
        print("\n".join(str(p) for p in ckpts))
        return

    # single category
    ckpt = run_one_category(args.category)
    if ckpt is None:
        raise SystemExit(2)
    print(str(ckpt))


if __name__ == "__main__":
    main()
