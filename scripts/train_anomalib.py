import os
os.environ["TQDM_DISABLE"] = "1"

# tqdm 강제 비활성화 (클래스 상속 유지하면서 disable=True 강제)
import tqdm
from tqdm import tqdm as tqdm_class

_original_tqdm_init = tqdm_class.__init__

def _patched_tqdm_init(self, *args, **kwargs):
    kwargs["disable"] = True
    _original_tqdm_init(self, *args, **kwargs)

tqdm_class.__init__ = _patched_tqdm_init
tqdm.tqdm = tqdm_class

import json
import time
from pathlib import Path

import torch
from anomalib.models import Patchcore, WinClip, EfficientAd
from anomalib.models.image.efficient_ad.torch_model import EfficientAdModelSize
from anomalib.engine import Engine
from pytorch_lightning.callbacks import Callback, EarlyStopping

# PyTorch 2.6+ weights_only=True 대응: Anomalib 클래스 허용
torch.serialization.add_safe_globals([EfficientAdModelSize])

from src.utils.loaders import load_config
from src.utils.log import setup_logger
from src.utils.device import get_device
from src.datasets.dataloader import MMADLoader

# Lazy Logger: 필요할 때만 생성
_train_logger = None
_inference_logger = None

def get_train_logger():
    global _train_logger
    if _train_logger is None:
        _train_logger = setup_logger(name="TrainAnomalib", log_prefix="train_anomalib")
    return _train_logger

def get_inference_logger():
    global _inference_logger
    if _inference_logger is None:
        _inference_logger = setup_logger(name="InferenceAnomalib", log_prefix="inference_anomalib")
    return _inference_logger

class EpochProgressCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs
        metrics = trainer.callback_metrics

        parts = [f"[Epoch {epoch}/{max_epochs}]"]

        # Loss
        train_loss = metrics.get("train_loss") or metrics.get("loss")
        if train_loss is not None:
            parts.append(f"loss={float(train_loss):.4f}")

        # AUROC
        auroc = metrics.get("image_AUROC") or metrics.get("AUROC")
        if auroc is not None:
            parts.append(f"AUROC={float(auroc):.4f}")

        # F1
        f1 = metrics.get("image_F1Score") or metrics.get("F1Score")
        if f1 is not None:
            parts.append(f"F1={float(f1):.4f}")

        print(" | ".join(parts), flush=True)


class EarlyStoppingTracker(Callback):
    """Early Stopping 이벤트를 추적하고 wandb에 기록하는 콜백."""

    def __init__(self, early_stopping_config: dict):
        super().__init__()
        self.config = early_stopping_config

    def on_train_end(self, trainer, pl_module):
        """학습 종료 시 early stopping 정보를 wandb에 기록."""
        import wandb

        # EarlyStopping 콜백에서 정보 추출
        early_stopping_cb = None
        for cb in trainer.callbacks:
            if isinstance(cb, EarlyStopping):
                early_stopping_cb = cb
                break

        if early_stopping_cb is None:
            return

        # Early stopping으로 종료되었는지 확인
        was_early_stopped = trainer.current_epoch < (trainer.max_epochs - 1)
        stopped_epoch = trainer.current_epoch + 1
        best_score = early_stopping_cb.best_score
        patience = self.config.get("patience", 10)

        # 콘솔 출력
        if was_early_stopped:
            print(f"\n⚡ Early Stopping triggered at epoch {stopped_epoch}")
            print(f"   Best {self.config.get('monitor', 'metric')}: {best_score:.4f}")
            get_train_logger().info(f"Early Stopping at epoch {stopped_epoch}, best score: {best_score:.4f}")

        # wandb에 기록 (2가지만)
        if wandb.run is not None:
            wandb.run.summary["early_stopping/patience_setting"] = patience  # 초기 patience 설정값
            wandb.run.summary["early_stopping/stopped_epoch"] = stopped_epoch  # 실제 종료된 epoch


class Anomalibs:
    def __init__(self, config_path: str = "configs/runtime.yaml"):
        self.config = load_config(config_path)
        self.model_name = self.config["anomaly"]["model"]
        self.model_params = self.filter_none(
            self.config["anomaly"].get(self.model_name, {})
        )
        self.training_config = self.filter_none(
            self.config.get("training", {})
        )
        self.data_root = Path(self.config["data"]["root"])
        self.output_root = Path(self.config["data"]["output_root"])
        self.output_config = self.config.get("output", {})
        self.engine_config = self.config.get("engine", {})
        self.device = get_device()
        self.loader = MMADLoader(config=self.config, model_name=self.model_name)
        print(f"Anomalibs initialized - model: {self.model_name}, device: {self.device}")

    @staticmethod
    def cleanup_memory():
        """GPU 및 시스템 메모리 캐시 강제 비활성화 및 정리"""
        import gc
        gc.collect() # Python 가비지 컬렉션
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # PyTorch CUDA 캐시 비움
            torch.cuda.ipc_collect() # IPC 메모리 비움

    @staticmethod
    def filter_none(d: dict) -> dict:
        return {k: v for k, v in d.items() if v is not None}

    # 모델별 checkpoint/early_stopping monitor 설정
    # - patchcore: 1 epoch만 학습, metric 로깅 없음 → monitor=None
    # - efficientad: iterative 학습, train_loss가 매 epoch 로깅됨
    # - winclip: zero-shot, 학습 없음
    MODEL_METRICS = {
        "patchcore": {"monitor": None, "mode": "max"},
        "efficientad": {"monitor": "train_loss", "mode": "min"},
        "winclip": {"monitor": None, "mode": "max"},
    }

    def get_evaluator(self):
        """val_metrics 포함 Evaluator 생성 (validation 시 메트릭 로깅용)"""
        from anomalib.metrics import AUROC, F1Score
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
        ]
        return Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)

    def get_model(self):
        if self.model_name == "patchcore":
            return Patchcore(**self.model_params)
        elif self.model_name == "winclip":
            return WinClip(**self.model_params)
        elif self.model_name == "efficientad":
            return EfficientAd(**self.model_params)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def get_datamodule_kwargs(self):
        # datamodule kwargs from training config
        kwargs = {}
        if "train_batch_size" in self.training_config:
            kwargs["train_batch_size"] = self.training_config["train_batch_size"]
        elif self.model_name == "efficientad":
            kwargs["train_batch_size"] = 1  # EfficientAd 1 필수
        if "eval_batch_size" in self.training_config:
            kwargs["eval_batch_size"] = self.training_config["eval_batch_size"]
        if "num_workers" in self.training_config:
            kwargs["num_workers"] = self.training_config["num_workers"]
        return kwargs

    def get_engine(self, dataset: str = None, category: str = None, model=None, datamodule=None, stage: str = None, version_dir: Path = None, is_resume: bool = False):
        # Anomalib의 ModelCheckpoint를 사용해야 _setup_anomalib_callbacks()에서 중복 추가 방지
        from anomalib.callbacks.checkpoint import ModelCheckpoint

        # WandB logger 설정 (predict 시에는 비활성화)
        logger_config = self.engine_config.get("logger", False)
        if logger_config == "wandb":
            if stage == "predict" or not (dataset and category):
                # predict 또는 dataset/category 없으면 wandb 비활성화
                logger_config = False
            else:
                from pytorch_lightning.loggers import WandbLogger
                from src.utils.wandbs import login_wandb
                login_wandb()
                import torch
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

                # batch_size 추출
                batch_size = self.training_config.get("train_batch_size")
                if batch_size is None and datamodule is not None:
                    batch_size = getattr(datamodule, "train_batch_size", None)
                if batch_size is None:
                    batch_size = 1 if self.model_name == "efficientad" else 32

                img_size = self.config.get("data", {}).get("image_size", (256, 256))
                max_epochs = self.training_config.get("max_epochs") or 100
                lr = getattr(model, "lr", None) if model else None
                weight_decay = getattr(model, "weight_decay", None) if model else None
                train_data = getattr(datamodule, "train_data", None)
                num_train_img = len(train_data) if train_data is not None else 0

                # run_name 조합: {dataset}_{category} 또는 {dataset}_{category}_{custom}
                wandb_config = self.config.get("wandb", {})
                custom_name = wandb_config.get("run_name")
                if custom_name:
                    run_name = f"{dataset}_{category}_{custom_name}"
                else:
                    run_name = f"{dataset}_{category}"

                # Early Stopping 설정 추출
                es_config = self.training_config.get("early_stopping", {})

                logger_config = WandbLogger(
                    project=wandb_config.get("project", "mmad-anomaly"),
                    name=run_name,
                    tags=[self.model_name, dataset, category],
                    config={
                        "model": self.model_name,
                        "dataset": dataset,
                        "category": category,
                        "image_size": img_size,
                        "num_train_img": num_train_img,
                        "device": gpu_name,
                        "batch_size": batch_size,
                        "max_epochs": max_epochs,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "early_stopping_patience": es_config.get("patience", 10) if es_config.get("enabled") else None,
                    },
                )

        enable_progress = self.engine_config.get("enable_progress_bar", False)
        callbacks = [] if enable_progress else [EpochProgressCallback()]

        # --- Custom Callbacks for Checkpoint and Visualizer ---

        # 1. ModelCheckpoint Callback - predict 시에는 불필요
        if dataset and category and stage != "predict" and version_dir:
            checkpoint_dir = version_dir
            # version_dir는 fit()에서 이미 생성됨

            monitor_cfg = self.MODEL_METRICS.get(
                self.model_name, {"monitor": "image_AUROC", "mode": "max"}
            )

            # PatchCore/WinCLIP: monitor=None이면 매 epoch 저장 (1 epoch만 학습)
            if monitor_cfg["monitor"] is None:
                model_checkpoint_callback = ModelCheckpoint(
                    dirpath=str(checkpoint_dir),
                    filename="model",
                    save_top_k=-1,  # 모든 checkpoint 저장
                    every_n_epochs=1,
                    auto_insert_metric_name=False,
                )
            else:
                # EfficientAd: metric 기반 best model 저장
                model_checkpoint_callback = ModelCheckpoint(
                    dirpath=str(checkpoint_dir),
                    filename="model",
                    save_last=False,
                    save_top_k=1,
                    monitor=monitor_cfg["monitor"],
                    mode=monitor_cfg["mode"],
                    auto_insert_metric_name=False,
                )
            callbacks.append(model_checkpoint_callback)
        # If dataset/category not available, default ModelCheckpoint might still be added by Anomalib.
        # Or, if this is a predict-only scenario without training, no checkpoint is needed.

        # Visualizer Callback (yaml에서 visualizer: true일 때만)
        visualizer_enabled = self.model_params.get("visualizer", False)
        if visualizer_enabled and stage == "predict" and dataset and category:
            try:
                # Anomalib 버전에 따라 import 경로가 다름
                try:
                    from anomalib.callbacks import ImageVisualizerCallback as VisualizerCallback
                except ImportError:
                    try:
                        from anomalib.utils.callbacks import ImageVisualizerCallback as VisualizerCallback
                    except ImportError:
                        VisualizerCallback = None

                if VisualizerCallback:
                    image_save_path = (
                        self.output_root
                        / self.MODEL_DIR_MAP.get(self.model_name, self.model_name.capitalize())
                        / dataset
                        / category
                        / "predictions"
                    )
                    image_save_path.mkdir(parents=True, exist_ok=True)
                    callbacks.append(VisualizerCallback(image_save_path=str(image_save_path)))
            except Exception as e:
                logger.warning(f"Visualizer callback not available: {e}")

        # 3. Early Stopping Callback (fit 시에만, 특정 모델만)
        # PatchCore: memory-bank 기반, 1 epoch만 학습 → early stopping 불필요
        # WinCLIP: zero-shot, 학습 없음 → early stopping 불필요
        # EfficientAd: iterative 학습 → early stopping 유용
        early_stop_config = self.training_config.get("early_stopping", {})
        models_need_early_stopping = ["efficientad"]  # early stopping이 유용한 모델 목록

        if (
            early_stop_config.get("enabled", False)
            and stage != "predict"
            and self.model_name in models_need_early_stopping
        ):
            # 모델별 메트릭 설정 사용 (checkpoint와 동일하게)
            model_metric = self.MODEL_METRICS.get(
                self.model_name, {"monitor": "image_AUROC", "mode": "max"}
            )
            es_monitor = model_metric["monitor"]
            es_mode = model_metric["mode"]

            early_stopping_callback = EarlyStopping(
                monitor=es_monitor,
                patience=early_stop_config.get("patience", 10),
                mode=es_mode,
                verbose=True,
                check_on_train_epoch_end=True,  # validation이 아닌 train epoch 후 체크
            )
            callbacks.append(early_stopping_callback)

            # Early Stopping 추적 콜백 추가 (wandb 로깅용)
            callbacks.append(EarlyStoppingTracker(early_stop_config))
            get_train_logger().info(
                f"Early Stopping enabled: monitor={es_monitor}, "
                f"patience={early_stop_config.get('patience')}, mode={es_mode}"
            )
        elif early_stop_config.get("enabled", False) and self.model_name not in models_need_early_stopping:
            get_train_logger().info(f"Early Stopping skipped: {self.model_name} doesn't need iterative training")

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

        # min_epochs 설정 (early stopping이 실제 적용되는 모델에만)
        if (
            early_stop_config.get("enabled", False)
            and early_stop_config.get("min_epochs")
            and self.model_name in models_need_early_stopping
        ):
            kwargs["min_epochs"] = early_stop_config["min_epochs"]

        return Engine(**kwargs)

    # Anomalib이 저장하는 실제 폴더명 매핑
    MODEL_DIR_MAP = {
        "patchcore": "Patchcore",
        "winclip": "WinClip",
        "efficientad": "EfficientAd",
    }

    def get_category_dir(self, dataset: str, category: str) -> Path:
        """카테고리 디렉토리 경로 반환."""
        model_dir = self.MODEL_DIR_MAP.get(self.model_name, self.model_name.capitalize())
        return self.output_root / model_dir / dataset / category

    def get_latest_version(self, dataset: str, category: str) -> int | None:
        """가장 최신 버전 번호 반환. 없으면 None."""
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
        """
        버전 디렉토리 경로 반환.
        - create_new=False (resume): 최신 버전 사용, 없으면 v0 생성
        - create_new=True (new training): 다음 버전 생성
        """
        category_dir = self.get_category_dir(dataset, category)
        latest = self.get_latest_version(dataset, category)

        if create_new:
            # 새 학습: 다음 버전 생성
            new_version = 0 if latest is None else latest + 1
            version_dir = category_dir / f"v{new_version}"
            version_dir.mkdir(parents=True, exist_ok=True)
            get_train_logger().info(f"Created new version directory: {version_dir}")
            return version_dir
        else:
            # Resume: 최신 버전 사용
            if latest is not None:
                return category_dir / f"v{latest}"
            else:
                # 버전 폴더 없으면 v0 생성
                version_dir = category_dir / "v0"
                version_dir.mkdir(parents=True, exist_ok=True)
                return version_dir

    def get_ckpt_path(self, dataset: str, category: str) -> Path | None:
        """최신 버전의 best 체크포인트 경로 반환.

        model-v2.ckpt > model-v1.ckpt > model.ckpt 순으로 최신 선택
        """
        if self.model_name == "winclip":
            return None

        latest = self.get_latest_version(dataset, category)
        if latest is None:
            return None

        version_dir = self.get_category_dir(dataset, category) / f"v{latest}"
        if not version_dir.exists():
            return None

        # model-v{n}.ckpt 파일들 찾기 (가장 높은 버전 선택)
        model_ckpts = list(version_dir.glob("model*.ckpt"))
        if not model_ckpts:
            return None

        # last.ckpt 제외하고, model로 시작하는 것만
        model_ckpts = [p for p in model_ckpts if p.name.startswith("model")]

        if not model_ckpts:
            return None

        # 버전 번호로 정렬 (model.ckpt=0, model-v1.ckpt=1, model-v2.ckpt=2)
        def get_version(path):
            name = path.stem  # model, model-v1, model-v2
            if name == "model":
                return 0
            elif "-v" in name:
                try:
                    return int(name.split("-v")[1])
                except ValueError:
                    return 0
            return 0

        best_ckpt = max(model_ckpts, key=get_version)
        return best_ckpt

    def requires_fit(self) -> bool:
        return self.model_name != "winclip"

    def fit(self, dataset: str, category: str):
        if not self.requires_fit():
            get_train_logger().info(f"{self.model_name} - no training required (zero-shot)")
            return self

        get_train_logger().info(f"Fitting {self.model_name} - {dataset}/{category}")

        # --- Resume/Version 관리 ---
        resume_training = self.training_config.get("resume", False)
        ckpt_path_to_use = None
        is_resume = False  # 실제로 이어서 학습하는지 여부

        if resume_training:
            # Resume: 최신 버전 폴더 사용, 체크포인트 있으면 이어서 학습
            version_dir = self.get_version_dir(dataset, category, create_new=False)
            potential_ckpt_path = self.get_ckpt_path(dataset, category)
            if potential_ckpt_path and potential_ckpt_path.exists():
                ckpt_path_to_use = str(potential_ckpt_path)
                is_resume = True
                get_train_logger().info(f"Resuming from: {ckpt_path_to_use}")
            else:
                get_train_logger().info(f"Resume enabled but no checkpoint found. Training in: {version_dir}")
        else:
            # New training: 새 버전 폴더 생성
            version_dir = self.get_version_dir(dataset, category, create_new=True)
            get_train_logger().info(f"New training in: {version_dir}")

        model = self.get_model()
        dm_kwargs = self.get_datamodule_kwargs()
        datamodule = self.loader.get_datamodule(dataset, category, **dm_kwargs)
        engine = self.get_engine(dataset, category, model=model, datamodule=datamodule, version_dir=version_dir, is_resume=is_resume)

        engine.fit(datamodule=datamodule, model=model, ckpt_path=ckpt_path_to_use)

        # WandB run 종료 (카테고리별로 별도 run)
        import wandb
        if wandb.run is not None:
            wandb.finish()

        # 메모리 해제
        del engine
        del model
        del datamodule
        self.cleanup_memory()

        get_train_logger().info(f"Fitting {dataset}/{category} done")
        return self

    def predict(self, dataset: str, category: str, save_json: bool = None, verbose: bool = True):
        if verbose:
            get_inference_logger().info(f"Predicting {self.model_name} - {dataset}/{category}")

        model = self.get_model()
        dm_kwargs = self.get_datamodule_kwargs()
        dm_kwargs["include_mask"] = True  # predict 시 GT mask 포함
        datamodule = self.loader.get_datamodule(dataset, category, **dm_kwargs)
        engine = self.get_engine(dataset, category, model=model, datamodule=datamodule, stage="predict")
        ckpt_path = self.get_ckpt_path(dataset, category)

        # WinCLIP requires class name for text embeddings
        if self.model_name == "winclip":
            model.setup(class_name=category)

        predictions = engine.predict(
            datamodule=datamodule,
            model=model,
            ckpt_path=ckpt_path,
        )

        # save json
        if save_json is None:
            save_json = self.output_config.get("save_json", False)
        if save_json:
            self.save_predictions_json(predictions, dataset, category)

        if verbose:
            get_inference_logger().info(f"Predicting {dataset}/{category} done - {len(predictions)} batches")
        return predictions

    def get_mask_path(self, image_path: str, dataset: str) -> str | None:
        """이미지 경로에서 대응하는 마스크 경로 추론"""
        image_path = Path(image_path)

        # GoodsAD: test/{defect_type}/xxx.jpg -> ground_truth/{defect_type}/xxx.png
        if dataset == "GoodsAD":
            parts = image_path.parts
            if "test" in parts:
                test_idx = parts.index("test")
                defect_type = parts[test_idx + 1]
                # good 폴더는 마스크 없음
                if defect_type == "good":
                    return None
                mask_path = (
                    image_path.parent.parent.parent
                    / "ground_truth"
                    / defect_type
                    / (image_path.stem + ".png")
                )
                if mask_path.exists():
                    return str(mask_path)
        # MVTec-AD, VisA, MVTec-LOCO: batch에 mask_path가 이미 있음
        return None

    def save_predictions_json(self, predictions, dataset: str, category: str):
        output_dir = self.output_root / "predictions" / self.model_name / dataset / category
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

                # 마스크 경로 추가 (batch에 있으면 사용, 없으면 추론)
                if "mask_path" in batch and batch["mask_path"][i]:
                    result["mask_path"] = str(batch["mask_path"][i])
                else:
                    mask_path = self.get_mask_path(image_path, dataset)
                    if mask_path:
                        result["mask_path"] = mask_path

                # ground truth label (정상/비정상)
                if "label" in batch:
                    result["gt_label"] = int(batch["label"][i])

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
        """Get list of (dataset, category) tuples from DATASETS."""
        return [
            (dataset, category)
            for dataset in self.loader.datasets_to_run
            for category in self.loader.get_categories(dataset)
        ]

    def get_trained_categories(self, filter_by_config: bool = True) -> list[tuple[str, str]]:
        """Get list of (dataset, category) tuples that have trained checkpoints.

        Args:
            filter_by_config: True면 YAML의 datasets 설정에 있는 것만 반환
        """
        model_dir = self.MODEL_DIR_MAP.get(self.model_name, self.model_name.capitalize())
        model_path = self.output_root / model_dir

        if not model_path.exists():
            return []

        # YAML에서 지정한 datasets
        config_datasets = set(self.loader.datasets_to_run) if filter_by_config else None

        trained = []
        for dataset_dir in sorted(model_path.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name

            # filter_by_config=True면 YAML에 있는 dataset만
            if config_datasets and dataset not in config_datasets:
                continue

            for category_dir in sorted(dataset_dir.iterdir()):
                if not category_dir.is_dir():
                    continue
                category = category_dir.name
                # 모든 버전 폴더에서 체크포인트 찾기
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
        get_train_logger().info(f"Starting fit_all: {total} categories")

        for idx, (dataset, category) in enumerate(categories, 1):
            msg_start = f"[{idx}/{total}] Training: {dataset}/{category}..."
            print(f"\n{msg_start}")
            get_train_logger().info(msg_start)
            start = time.time()
            self.fit(dataset, category)
            elapsed = time.time() - start
            msg_done = f"[{idx}/{total}] {dataset}/{category} done ({elapsed:.1f}s)"
            print(f"✓ {msg_done}")
            get_train_logger().info(msg_done)

        get_train_logger().info(f"fit_all completed: {total} categories")

    def predict_all(self, save_json: bool = None):
        categories = self.get_trained_categories()
        total = len(categories)
        get_inference_logger().info(f"Starting predict_all: {total} trained categories")

        all_predictions = {}
        for idx, (dataset, category) in enumerate(categories, 1):
            msg_start = f"[{idx}/{total}] Inference: {dataset}/{category}..."
            print(f"\n{msg_start}")
            get_inference_logger().info(msg_start)
            start = time.time()
            key = f"{dataset}/{category}"
            all_predictions[key] = self.predict(dataset, category, save_json)
            elapsed = time.time() - start
            msg_done = f"[{idx}/{total}] {dataset}/{category} done ({elapsed:.1f}s)"
            print(f"✓ {msg_done}")
            get_inference_logger().info(msg_done)

        get_inference_logger().info(f"predict_all completed: {total} categories")
        return all_predictions
