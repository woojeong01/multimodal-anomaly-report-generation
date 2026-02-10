"""PatchCore 학습 스크립트

Usage:
    python scripts/train.py --config configs/anomaly.yaml
    python scripts/train.py --config configs/anomaly.yaml --normal-dir /path/to/good
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from anomalib.models import Patchcore
from anomalib.data import Folder
from anomalib.engine import Engine

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, setup_logger, get_device

logger = setup_logger(name="Train", log_prefix="train")

# PatchCore 기본값
DEFAULTS = {
    "backbone": "wide_resnet50_2",
    "layers": ["layer2", "layer3"],
    "image_size": (1024, 1024),
    "train_batch_size": 32,
    "eval_batch_size": 32,
    "num_workers": 4,
    "output_dir": "checkpoints",
}


def get_value(config: dict, key: str, default):
    """config에서 값 가져오기 (null이면 default 반환)"""
    value = config.get(key)
    return value if value is not None else default


def get_system_info() -> dict:
    """시스템 정보 수집 (wandb 로깅용)"""
    info = {
        "device": str(get_device()),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["cuda_version"] = torch.version.cuda
    return info


def init_wandb(config: dict, model_params: dict):
    """wandb 초기화"""
    wandb_config = config.get("wandb", {})
    if not wandb_config.get("enabled", False):
        return None

    try:
        import wandb
        system_info = get_system_info()

        run = wandb.init(
            project=wandb_config.get("project", "MMAD_Inspector"),
            name=wandb_config.get("run_name"),
            tags=wandb_config.get("tags", []),
            config={
                "model": model_params,
                "training": config.get("training", {}),
                "system": system_info,
            },
        )
        logger.info(f"wandb initialized: {run.url}")
        return run
    except ImportError:
        logger.warning("wandb not installed. Skipping.")
        return None


def train_patchcore(
    normal_image_paths: list[str],
    output_dir: str,
    category: str = "custom",
    backbone: str = DEFAULTS["backbone"],
    layers: list[str] = DEFAULTS["layers"],
    image_size: tuple[int, int] = DEFAULTS["image_size"],
    train_batch_size: int = DEFAULTS["train_batch_size"],
    eval_batch_size: int = DEFAULTS["eval_batch_size"],
    num_workers: int = DEFAULTS["num_workers"],
    wandb_run=None,
) -> str:
    """PatchCore 모델 학습 (Memory Bank 구축)"""
    logger.info(f"Training: {len(normal_image_paths)} images, category={category}")
    logger.info(f"Model: backbone={backbone}, layers={layers}, image_size={image_size}")

    model = Patchcore(
        backbone=backbone,
        layers=layers,
        pre_trained=True,
    )

    with TemporaryDirectory() as tmpdir:
        normal_dir = Path(tmpdir) / "train" / "good"
        normal_dir.mkdir(parents=True)

        for i, img_path in enumerate(normal_image_paths):
            ext = Path(img_path).suffix
            shutil.copy(img_path, normal_dir / f"{i:04d}{ext}")

        datamodule = Folder(
            name=category,
            root=tmpdir,
            normal_dir="train/good",
            image_size=image_size,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
        )
        datamodule.setup()

        engine = Engine(
            task="segmentation",
            default_root_dir=output_dir,
            max_epochs=1,
        )
        engine.fit(model=model, datamodule=datamodule)

    ckpt_path = Path(output_dir) / "weights" / "lightning" / "model.ckpt"
    logger.info(f"Checkpoint saved: {ckpt_path}")

    if wandb_run:
        import wandb
        wandb.log({"checkpoint_path": str(ckpt_path)})
        wandb.finish()

    return str(ckpt_path)


def get_image_paths(data_dir: str, extensions: list[str]) -> list[str]:
    """디렉토리에서 이미지 경로 수집"""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    image_paths = []
    for ext in extensions:
        image_paths.extend(data_path.glob(f"**/*{ext}"))
        image_paths.extend(data_path.glob(f"**/*{ext.upper()}"))

    return [str(p) for p in sorted(image_paths)]


def main():
    parser = argparse.ArgumentParser(description="Train PatchCore model")
    parser.add_argument("--config", type=str, default="configs/anomaly.yaml")
    parser.add_argument("--normal-dir", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--category", type=str, default="custom")
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = config.get("anomaly", {}).get("patchcore", {})
    training = config.get("training", {})
    data = config.get("data", {})

    # 파라미터 (CLI > Config > Default)
    normal_dir = args.normal_dir or get_value(data, "normal_dir", None)
    if not normal_dir:
        raise ValueError("--normal-dir required or set data.normal_dir in config")

    output_dir = args.output_dir or get_value(training, "output_dir", DEFAULTS["output_dir"])

    # 모델 파라미터 (null이면 기본값)
    model_params = {
        "backbone": get_value(model_config, "backbone", DEFAULTS["backbone"]),
        "layers": get_value(model_config, "layers", DEFAULTS["layers"]),
        "image_size": tuple(get_value(model_config, "image_size", list(DEFAULTS["image_size"]))),
    }

    # 학습 파라미터
    train_params = {
        "train_batch_size": get_value(training, "train_batch_size", DEFAULTS["train_batch_size"]),
        "eval_batch_size": get_value(training, "eval_batch_size", DEFAULTS["eval_batch_size"]),
        "num_workers": get_value(training, "num_workers", DEFAULTS["num_workers"]),
    }

    # wandb 초기화
    wandb_run = init_wandb(config, model_params)

    # 이미지 수집
    extensions = get_value(data, "image_extensions", [".jpg", ".jpeg", ".png"])
    normal_images = get_image_paths(normal_dir, extensions)
    logger.info(f"Found {len(normal_images)} images in {normal_dir}")

    if len(normal_images) == 0:
        raise ValueError(f"No images found in {normal_dir}")

    # 학습
    ckpt_path = train_patchcore(
        normal_image_paths=normal_images,
        output_dir=output_dir,
        category=args.category,
        wandb_run=wandb_run,
        **model_params,
        **train_params,
    )

    print(f"\nCheckpoint: {ckpt_path}")
    print(f"Update config: anomaly.patchcore.checkpoint_path: {ckpt_path}")


if __name__ == "__main__":
    main()
