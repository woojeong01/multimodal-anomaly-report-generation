"""PatchCore Training and Evaluation Script using Anomalib.

Usage:
    # Train all categories
    python scripts/train_anomalib.py --mode fit

    # Train specific category
    python scripts/train_anomalib.py --mode fit --dataset GoodsAD --category cigarette_box

    # Evaluate (test) all trained models
    python scripts/train_anomalib.py --mode test

    # Predict
    python scripts/train_anomalib.py --mode predict
"""
import os
os.environ["TQDM_DISABLE"] = "1"

import tqdm
from tqdm import tqdm as tqdm_class

_original_tqdm_init = tqdm_class.__init__

def _patched_tqdm_init(self, *args, **kwargs):
    kwargs["disable"] = True
    _original_tqdm_init(self, *args, **kwargs)

tqdm_class.__init__ = _patched_tqdm_init
tqdm.tqdm = tqdm_class

import argparse
import json
import time
from pathlib import Path

import torch
from anomalib.models import Patchcore
from anomalib.engine import Engine
from pytorch_lightning.callbacks import Callback

from src.utils.loaders import load_config
from src.utils.log import setup_logger
from src.utils.device import get_device
from src.datasets.dataloader import MMADLoader

import logging as _logging
_logging.getLogger("pytorch_lightning").setLevel(_logging.WARNING)
_logging.getLogger("lightning.pytorch").setLevel(_logging.WARNING)

_train_logger = None
_inference_logger = None

def get_train_logger():
    global _train_logger
    if _train_logger is None:
        _train_logger = setup_logger(name="TrainAnomalib", log_prefix="train_anomalib", console_logging=False)
    return _train_logger

def get_inference_logger():
    global _inference_logger
    if _inference_logger is None:
        _inference_logger = setup_logger(name="InferenceAnomalib", log_prefix="inference_anomalib", console_logging=False)
    return _inference_logger


class EpochProgressCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs
        metrics = trainer.callback_metrics

        parts = [f"[Epoch {epoch}/{max_epochs}]"]

        train_loss = metrics.get("train_loss") or metrics.get("loss")
        if train_loss is not None:
            parts.append(f"loss={float(train_loss):.4f}")

        auroc = metrics.get("image_AUROC") or metrics.get("AUROC")
        if auroc is not None:
            parts.append(f"AUROC={float(auroc):.4f}")

        f1 = metrics.get("image_F1Score") or metrics.get("F1Score")
        if f1 is not None:
            parts.append(f"F1={float(f1):.4f}")

        print(" | ".join(parts), flush=True)


class PatchCoreTrainer:
    """PatchCore model trainer using Anomalib."""

    MODEL_DIR = "Patchcore"

    def __init__(self, config_path: str = "configs/anomaly.yaml"):
        self.config = load_config(config_path)
        self.model_params = self._filter_none(
            self.config["anomaly"].get("patchcore", {})
        )
        self.training_config = self._filter_none(
            self.config.get("training", {})
        )
        self.data_root = Path(self.config["data"]["root"])
        self.output_root = Path(self.config["data"]["output_root"])
        self.output_config = self.config.get("output", {})
        self.predict_config = self.config.get("predict", {})
        self.engine_config = self.config.get("engine", {})
        self.device = get_device()
        self.accelerator = self.engine_config.get("accelerator", "auto")
        self.loader = MMADLoader(config=self.config, model_name="patchcore")
        print(f"[PatchCore] device: {self.device}, accelerator: {self.accelerator}")

    @staticmethod
    def _filter_none(d: dict) -> dict:
        return {k: v for k, v in d.items() if v is not None}

    @staticmethod
    def cleanup_memory():
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def get_evaluator(self):
        """Create Evaluator with metrics including AUPRO."""
        from anomalib.metrics import AUROC, F1Score, AUPRO
        from anomalib.metrics.evaluator import Evaluator

        val_metrics = [
            AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
            F1Score(fields=["pred_label", "gt_label"], prefix="image_", strict=False),
        ]
        test_metrics = [
            AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
            F1Score(fields=["pred_label", "gt_label"], prefix="image_"),
            AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
            F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_", strict=False),
            AUPRO(fields=["anomaly_map", "gt_mask"], strict=False),
        ]
        return Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)

    def get_model(self, with_evaluator: bool = True):
        """Create PatchCore model with optional evaluator.

        Args:
            with_evaluator: If True, attach evaluator with AUPRO metrics
        """
        if with_evaluator:
            evaluator = self.get_evaluator()
            return Patchcore(evaluator=evaluator, **self.model_params)
        return Patchcore(**self.model_params)

    def get_datamodule_kwargs(self):
        kwargs = {}
        if "train_batch_size" in self.training_config:
            kwargs["train_batch_size"] = self.training_config["train_batch_size"]
        if "eval_batch_size" in self.training_config:
            kwargs["eval_batch_size"] = self.training_config["eval_batch_size"]
        if "num_workers" in self.training_config:
            kwargs["num_workers"] = self.training_config["num_workers"]
        return kwargs

    def get_category_dir(self, dataset: str, category: str) -> Path:
        return self.output_root / self.MODEL_DIR / dataset / category

    def get_latest_version(self, dataset: str, category: str) -> int | None:
        category_dir = self.get_category_dir(dataset, category)
        if not category_dir.exists():
            return None

        versions = []
        for d in category_dir.iterdir():
            if d.is_dir() and d.name.startswith("v"):
                try:
                    versions.append(int(d.name[1:]))
                except ValueError:
                    continue
        return max(versions) if versions else None

    def get_version_dir(self, dataset: str, category: str, create_new: bool = False) -> Path:
        category_dir = self.get_category_dir(dataset, category)
        latest = self.get_latest_version(dataset, category)

        if create_new:
            new_version = 0 if latest is None else latest + 1
            version_dir = category_dir / f"v{new_version}"
            get_train_logger().info(f"New version directory: {version_dir}")
            return version_dir
        else:
            if latest is not None:
                return category_dir / f"v{latest}"
            else:
                version_dir = category_dir / "v0"
                version_dir.mkdir(parents=True, exist_ok=True)
                return version_dir

    def get_ckpt_path(self, dataset: str, category: str) -> Path | None:
        target_version = self.predict_config.get("version", None)

        if target_version is not None:
            version_dir = self.get_category_dir(dataset, category) / f"v{target_version}"
            if not version_dir.exists():
                get_train_logger().warning(
                    f"Specified version v{target_version} not found for {dataset}/{category}. "
                    f"Falling back to latest version."
                )
                target_version = None

        if target_version is None:
            latest = self.get_latest_version(dataset, category)
            if latest is None:
                return None
            version_dir = self.get_category_dir(dataset, category) / f"v{latest}"

        if not version_dir.exists():
            return None

        model_ckpts = list(version_dir.glob("model*.ckpt"))
        if not model_ckpts:
            return None

        model_ckpts = [p for p in model_ckpts if p.name.startswith("model")]
        if not model_ckpts:
            return None

        def get_version(path):
            name = path.stem
            if name == "model":
                return 0
            elif "-v" in name:
                try:
                    return int(name.split("-v")[1])
                except ValueError:
                    return 0
            return 0

        return max(model_ckpts, key=get_version)

    def get_engine(self, dataset: str = None, category: str = None, version_dir: Path = None, stage: str = None):
        from anomalib.callbacks.checkpoint import ModelCheckpoint

        logger_config = self.engine_config.get("logger", False)
        if logger_config == "wandb" and stage not in ["predict", "test"]:
            from pytorch_lightning.loggers import WandbLogger
            from src.utils.wandbs import login_wandb
            login_wandb()

            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            img_size = self.config.get("data", {}).get("image_size", (256, 256))
            wandb_config = self.config.get("wandb", {})
            custom_name = wandb_config.get("run_name")
            run_name = f"{dataset}_{category}_{custom_name}" if custom_name else f"{dataset}_{category}"

            raw_model_params = self.config["anomaly"].get("patchcore", {})
            model_hparams = {
                "coreset_sampling_ratio": raw_model_params.get("coreset_sampling_ratio") or 0.1
            }

            logger_config = WandbLogger(
                project=wandb_config.get("project", "mmad-anomaly"),
                name=run_name,
                tags=["patchcore", dataset, category],
                config={
                    "model": "patchcore",
                    "dataset": dataset,
                    "category": category,
                    "image_size": img_size,
                    "device": gpu_name,
                    **model_hparams,
                },
            )
        elif stage in ["predict", "test"]:
            logger_config = False

        enable_progress = self.engine_config.get("enable_progress_bar", False)
        callbacks = [] if enable_progress else [EpochProgressCallback()]

        if dataset and category and stage not in ["predict", "test"] and version_dir:
            model_checkpoint_callback = ModelCheckpoint(
                dirpath=str(version_dir),
                filename="model",
                save_top_k=-1,
                every_n_epochs=1,
                auto_insert_metric_name=False,
            )
            callbacks.append(model_checkpoint_callback)

        kwargs = {
            "accelerator": self.engine_config.get("accelerator", "auto"),
            "devices": 1,
            "default_root_dir": str(self.output_root),
            "logger": logger_config,
            "enable_progress_bar": enable_progress,
            "callbacks": callbacks,
        }

        if "max_epochs" in self.training_config:
            kwargs["max_epochs"] = self.training_config["max_epochs"]

        return Engine(**kwargs)

    def fit(self, dataset: str, category: str):
        resume_training = self.training_config.get("resume", False)
        ckpt_path_to_use = None

        if resume_training:
            version_dir = self.get_version_dir(dataset, category, create_new=False)
            potential_ckpt_path = self.get_ckpt_path(dataset, category)
            if potential_ckpt_path and potential_ckpt_path.exists():
                ckpt_path_to_use = str(potential_ckpt_path)
                get_train_logger().info(f"Resuming from: {ckpt_path_to_use}")
            else:
                get_train_logger().info(f"Resume enabled but no checkpoint found. Training in: {version_dir}")
        else:
            version_dir = self.get_version_dir(dataset, category, create_new=True)
            get_train_logger().info(f"New training in: {version_dir}")

        model = self.get_model()
        dm_kwargs = self.get_datamodule_kwargs()
        datamodule = self.loader.get_datamodule(dataset, category, **dm_kwargs)
        engine = self.get_engine(dataset, category, version_dir=version_dir)

        engine.fit(datamodule=datamodule, model=model, ckpt_path=ckpt_path_to_use)

        import wandb
        if wandb.run is not None:
            wandb.finish()

        del engine, model, datamodule
        self.cleanup_memory()

        return self

    def test(self, dataset: str, category: str) -> dict:
        """Evaluate model on test set. Returns metrics dict."""
        model = self.get_model()
        dm_kwargs = self.get_datamodule_kwargs()
        dm_kwargs["include_mask"] = True
        datamodule = self.loader.get_datamodule(dataset, category, **dm_kwargs)
        ckpt_path = self.get_ckpt_path(dataset, category)

        if ckpt_path is None:
            print(f"  No checkpoint found for {dataset}/{category}")
            return {}

        engine = self.get_engine(dataset, category, stage="test")
        results = engine.test(datamodule=datamodule, model=model, ckpt_path=ckpt_path)

        del engine, model, datamodule
        self.cleanup_memory()

        if results and len(results) > 0:
            return results[0]
        return {}

    def predict(self, dataset: str, category: str, save_json: bool = None):
        model = self.get_model()
        dm_kwargs = self.get_datamodule_kwargs()
        dm_kwargs["include_mask"] = True
        datamodule = self.loader.get_datamodule(dataset, category, **dm_kwargs)
        ckpt_path = self.get_ckpt_path(dataset, category)

        engine = self.get_engine(dataset, category, stage="predict")
        predictions = engine.predict(datamodule=datamodule, model=model, ckpt_path=ckpt_path)

        if save_json is None:
            save_json = self.output_config.get("save_json", False)
        if save_json:
            self._save_predictions_json(predictions, dataset, category)

        return predictions

    def _save_predictions_json(self, predictions, dataset: str, category: str):
        target_version = self.predict_config.get("version", None)
        if target_version is not None:
            version_tag = f"v{target_version}"
        else:
            latest = self.get_latest_version(dataset, category)
            version_tag = f"v{latest}" if latest is not None else "v0"

        output_dir = self.output_root / "predictions" / "patchcore" / dataset / category / version_tag
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for batch in predictions:
            for i in range(len(batch["image_path"])):
                image_path = str(batch["image_path"][i])
                result = {
                    "image_path": image_path,
                    "pred_score": float(batch["pred_score"][i]),
                    "pred_label": int(batch["pred_label"][i]),
                }

                if "anomaly_map" in batch and batch["anomaly_map"] is not None:
                    amap = batch["anomaly_map"][i]
                    result["anomaly_map_shape"] = list(amap.shape)
                    result["anomaly_map_max"] = float(amap.max())
                    result["anomaly_map_mean"] = float(amap.mean())

                results.append(result)

        json_path = output_dir / "predictions.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        get_inference_logger().info(f"Saved predictions JSON: {json_path}")

    def get_all_categories(self) -> list[tuple[str, str]]:
        config_categories = self.config.get("data", {}).get("categories", None)
        all_cats = [
            (dataset, category)
            for dataset in self.loader.datasets_to_run
            for category in self.loader.get_categories(dataset)
        ]
        if config_categories:
            all_cats = [(d, c) for d, c in all_cats if c in config_categories]
        return all_cats

    def get_trained_categories(self, filter_by_config: bool = True) -> list[tuple[str, str]]:
        model_path = self.output_root / self.MODEL_DIR

        if not model_path.exists():
            return []

        config_datasets = set(self.loader.datasets_to_run) if filter_by_config else None
        config_categories = set(self.config.get("data", {}).get("categories", []) or [])

        trained = []
        for dataset_dir in sorted(model_path.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name

            if config_datasets and dataset not in config_datasets:
                continue

            for category_dir in sorted(dataset_dir.iterdir()):
                if not category_dir.is_dir():
                    continue
                category = category_dir.name

                if config_categories and category not in config_categories:
                    continue

                has_checkpoint = False
                for version_dir in category_dir.iterdir():
                    if version_dir.is_dir() and version_dir.name.startswith("v"):
                        if (version_dir / "model.ckpt").exists():
                            has_checkpoint = True
                            break
                if has_checkpoint:
                    trained.append((dataset, category))
        return trained

    def fit_all(self):
        categories = self.get_all_categories()
        total = len(categories)
        get_train_logger().info(f"fit_all: {total} categories")

        for idx, (dataset, category) in enumerate(categories, 1):
            print(f"\n[{idx}/{total}] Training: {dataset}/{category}...")
            start = time.time()
            self.fit(dataset, category)
            elapsed = time.time() - start
            msg = f"[{idx}/{total}] {dataset}/{category} done ({elapsed:.1f}s)"
            print(f"  {msg}")
            get_train_logger().info(msg)

        get_train_logger().info(f"fit_all completed: {total} categories")

    def test_all(self) -> dict:
        """Evaluate all trained models. Returns dict of metrics."""
        categories = self.get_trained_categories()
        total = len(categories)
        print(f"\nEvaluating {total} trained models...")

        all_results = {}
        for idx, (dataset, category) in enumerate(categories, 1):
            key = f"{dataset}/{category}"
            print(f"\n[{idx}/{total}] Testing: {key}...")
            start = time.time()
            metrics = self.test(dataset, category)
            elapsed = time.time() - start

            all_results[key] = metrics

            # Print metrics
            img_auroc = metrics.get("image_AUROC", 0)
            pixel_auroc = metrics.get("pixel_AUROC", 0)
            aupro = metrics.get("AUPRO", 0)
            print(f"  Image AUROC: {img_auroc:.4f} | Pixel AUROC: {pixel_auroc:.4f} | AUPRO: {aupro:.4f} ({elapsed:.1f}s)")

        # Print summary
        if all_results:
            print("\n" + "=" * 70)
            print(f"{'Category':<35} {'I-AUROC':>10} {'P-AUROC':>10} {'AUPRO':>10}")
            print("=" * 70)
            for key, metrics in all_results.items():
                print(f"{key:<35} {metrics.get('image_AUROC', 0):>10.4f} "
                      f"{metrics.get('pixel_AUROC', 0):>10.4f} {metrics.get('AUPRO', 0):>10.4f}")
            print("=" * 70)

            # Average
            avg_img = sum(m.get("image_AUROC", 0) for m in all_results.values()) / len(all_results)
            avg_pix = sum(m.get("pixel_AUROC", 0) for m in all_results.values()) / len(all_results)
            avg_pro = sum(m.get("AUPRO", 0) for m in all_results.values()) / len(all_results)
            print(f"{'Average':<35} {avg_img:>10.4f} {avg_pix:>10.4f} {avg_pro:>10.4f}")

        return all_results

    def predict_all(self, save_json: bool = None):
        categories = self.get_trained_categories()
        total = len(categories)
        get_inference_logger().info(f"predict_all: {total} trained categories")

        all_predictions = {}
        for idx, (dataset, category) in enumerate(categories, 1):
            print(f"\n[{idx}/{total}] Predicting: {dataset}/{category}...")
            start = time.time()
            key = f"{dataset}/{category}"
            all_predictions[key] = self.predict(dataset, category, save_json)
            elapsed = time.time() - start
            msg = f"[{idx}/{total}] {dataset}/{category} done ({elapsed:.1f}s)"
            print(f"  {msg}")
            get_inference_logger().info(msg)

        get_inference_logger().info(f"predict_all completed: {total} categories")
        return all_predictions


def main():
    parser = argparse.ArgumentParser(description="PatchCore Training/Evaluation")
    parser.add_argument("--config", type=str, default="configs/anomaly.yaml")
    parser.add_argument("--mode", type=str, default="fit", choices=["fit", "test", "predict"])
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--save-json", action="store_true")
    args = parser.parse_args()

    trainer = PatchCoreTrainer(config_path=args.config)

    if args.mode == "fit":
        if args.dataset and args.category:
            trainer.fit(args.dataset, args.category)
        else:
            trainer.fit_all()
    elif args.mode == "test":
        if args.dataset and args.category:
            metrics = trainer.test(args.dataset, args.category)
            print(f"Results: {metrics}")
        else:
            trainer.test_all()
    elif args.mode == "predict":
        if args.dataset and args.category:
            trainer.predict(args.dataset, args.category, save_json=args.save_json)
        else:
            trainer.predict_all(save_json=args.save_json)


if __name__ == "__main__":
    main()
