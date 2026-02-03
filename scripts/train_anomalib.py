from pathlib import Path
import json

from anomalib.models import Patchcore, WinClip, EfficientAd
from anomalib.engine import Engine

from src.utils.loaders import load_config
from src.utils.log import setup_logger
from src.utils.device import get_device
from src.datasets.dataloader import MMADLoader

logger = setup_logger(name="TrainAnomalib", log_prefix="train_anomalib")


class Anomalibs:
    def __init__(self, config_path: str = "configs/runtime.yaml"):
        self.config = load_config(config_path)

        # model
        self.model_name = self.config["anomaly"]["model"]
        self.model_params = self.filter_none(
            self.config["anomaly"].get(self.model_name, {})
        )

        # training
        self.training_config = self.filter_none(
            self.config.get("training", {})
        )

        # data
        self.data_root = Path(self.config["data"]["root"])
        self.output_root = Path(self.config["data"]["output_root"])

        # engine
        self.output_config = self.config.get("output", {})
        self.engine_config = self.config.get("engine", {})

        # device (for logging)
        self.device = get_device()

        # MMAD loader
        self.loader = MMADLoader(root=str(self.data_root))

        logger.info(f"Initialized - model: {self.model_name}, device: {self.device}")

    @staticmethod
    def filter_none(d: dict) -> dict:
        return {k: v for k, v in d.items() if v is not None}

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
        if "eval_batch_size" in self.training_config:
            kwargs["eval_batch_size"] = self.training_config["eval_batch_size"]
        if "num_workers" in self.training_config:
            kwargs["num_workers"] = self.training_config["num_workers"]
        return kwargs

    def get_engine(self):
        kwargs = {
            "accelerator": self.engine_config.get("accelerator", "auto"),
            "devices": 1,
            "default_root_dir": str(self.output_root),
            "logger": self.engine_config.get("logger", False),
            "enable_progress_bar": self.engine_config.get("enable_progress_bar", False),
        }

        if "max_epochs" in self.training_config:
            kwargs["max_epochs"] = self.training_config["max_epochs"]
        return Engine(**kwargs)

    def get_ckpt_path(self, dataset: str, category: str) -> Path | None:
        if self.model_name == "winclip":
            return None
        return (
            self.output_root
            / self.model_name.capitalize()
            / dataset
            / category
            / "v0/weights/lightning/model.ckpt"
        )

    def requires_fit(self) -> bool:
        return self.model_name != "winclip"

    def fit(self, dataset: str, category: str):
        if not self.requires_fit():
            logger.info(f"{self.model_name} - no training required (zero-shot)")
            return self

        logger.info(f"Fitting {self.model_name} - {dataset}/{category}")

        model = self.get_model()
        dm_kwargs = self.get_datamodule_kwargs()
        datamodule = self.loader.get_datamodule(dataset, category, **dm_kwargs)
        engine = self.get_engine()

        engine.fit(datamodule=datamodule, model=model)
        logger.info(f"Fitting {dataset}/{category} done")

        return self

    def predict(self, dataset: str, category: str, save_json: bool = None):
        logger.info(f"Predicting {self.model_name} - {dataset}/{category}")

        model = self.get_model()
        dm_kwargs = self.get_datamodule_kwargs()
        datamodule = self.loader.get_datamodule(dataset, category, **dm_kwargs)
        engine = self.get_engine()
        ckpt_path = self.get_ckpt_path(dataset, category)

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

        logger.info(f"Predicting {dataset}/{category} done - {len(predictions)} batches")
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

        logger.info(f"Saved predictions JSON: {json_path}")

    def get_all_categories(self) -> list[tuple[str, str]]:
        """Get list of (dataset, category) tuples."""
        return [
            (dataset, category)
            for dataset in self.loader.DATASETS
            for category in self.loader.get_categories(dataset)
        ]

    def fit_all(self):
        categories = self.get_all_categories()
        total = len(categories)
        logger.info(f"Starting fit_all: {total} categories")

        for idx, (dataset, category) in enumerate(categories, 1):
            logger.info(f"[{idx}/{total}] {dataset}/{category}")
            self.fit(dataset, category)

        logger.info(f"fit_all completed: {total} categories")

    def predict_all(self, save_json: bool = None):
        categories = self.get_all_categories()
        total = len(categories)
        logger.info(f"Starting predict_all: {total} categories")

        all_predictions = {}
        for idx, (dataset, category) in enumerate(categories, 1):
            logger.info(f"[{idx}/{total}] {dataset}/{category}")
            key = f"{dataset}/{category}"
            all_predictions[key] = self.predict(dataset, category, save_json)

        logger.info(f"predict_all completed: {total} categories")
        return all_predictions
