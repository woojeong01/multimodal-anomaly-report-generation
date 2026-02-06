from pathlib import Path
from typing import Generator
import pandas as pd
import torch
import numpy as np
from PIL import Image
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from anomalib.data import MVTecAD, Visa
from anomalib.data.datamodules.base import AnomalibDataModule
from anomalib.data.datasets.base.image import AnomalibDataset
from anomalib.data.dataclasses import ImageBatch, ImageItem
from anomalib.data.utils.image import read_image, read_mask


def collate_items(items: list) -> ImageBatch:
    """ImageItem 리스트를 ImageBatch로 변환하는 collate 함수."""
    # items가 리스트가 아닌 경우 처리
    if not isinstance(items, (list, tuple)):
        items = [items]

    if len(items) == 0:
        raise ValueError("Empty batch received in collate_items")

    # 각 필드를 수집하여 배치 생성
    image_paths = [item.image_path for item in items]
    images = torch.stack([item.image for item in items])
    gt_labels = torch.stack([item.gt_label for item in items])

    # gt_mask 처리 - 모두 None이 아니면 스택, dtype 통일
    gt_masks = None
    if all(item.gt_mask is not None for item in items):
        gt_masks = torch.stack([item.gt_mask.float() for item in items])

    return ImageBatch(
        image_path=image_paths,
        image=images,
        gt_label=gt_labels,
        gt_mask=gt_masks,
    )


class MVTecLOCODataset(AnomalibDataset):
    """MVTec-LOCO 커스텀 Dataset (중첩 마스크 구조 처리)"""
    def __init__(
        self,
        root: Path | str,
        category: str,
        split: str = "train",
        preprocess=None,
        image_size: tuple[int, int] = (256, 256),
    ):
        super().__init__(augmentations=None)
        self.root = Path(root)
        self._category = category
        self.split = split
        self._samples = self.make_dataset()
        self.preprocess = preprocess  # 통합 전처리(transform/augmentation 등)
        self.image_size = image_size

    def make_dataset(self) -> pd.DataFrame:
        """MVTec-LOCO 샘플 DataFrame 생성"""
        cat_path = self.root / self.category
        samples_list = []

        if self.split == "train":
            train_good = cat_path / "train" / "good"
            if train_good.exists():
                for img_path in sorted(train_good.glob("*.png")):
                    if img_path.name.startswith("."):
                        continue
                    samples_list.append({
                        "image_path": str(img_path),
                        "label": "normal",
                        "label_index": 0,
                        "mask_path": None,
                        "split": "train",
                    })
        else:
            test_dir = cat_path / "test"
            gt_dir = cat_path / "ground_truth"

            if test_dir.exists():
                for defect_dir in sorted(test_dir.iterdir()):
                    if not defect_dir.is_dir() or defect_dir.name.startswith("."):
                        continue

                    defect_type = defect_dir.name
                    is_normal = defect_type == "good"

                    for img_path in sorted(defect_dir.glob("*.png")):
                        if img_path.name.startswith("."):
                            continue

                        mask_path = None
                        if not is_normal:
                            stem = img_path.stem
                            mask_dir = gt_dir / defect_type / stem
                            # MVTec-LOCO: 마스크 폴더 내 파일명은 000.png, 001.png, ... (이미지 stem과 다름)
                            if mask_dir.exists() and mask_dir.is_dir():
                                mask_files = sorted(mask_dir.glob("*.png"))
                                if mask_files:
                                    # 첫 번째 마스크 사용 (여러 개면 나중에 합칠 수 있음)
                                    mask_path = str(mask_files[0])

                        samples_list.append({
                            "image_path": str(img_path),
                            "label": "normal" if is_normal else "anomaly",
                            "label_index": 0 if is_normal else 1,
                            "mask_path": mask_path,
                            "split": "test",
                        })

        samples = pd.DataFrame(samples_list)
        samples.attrs["task"] = "segmentation"
        return samples

    def __getitem__(self, index: int) -> ImageItem:
        sample = self.samples.iloc[index]
        image_np = read_image(sample.image_path)

        # PIL.Image.fromarray가 float64 타입을 처리 못하므로 uint8로 변환
        if image_np.dtype in [np.float32, np.float64]:
            image_np = (image_np * 255).astype(np.uint8)
        image = Image.fromarray(image_np)

        gt_mask = None
        if pd.notna(sample.mask_path) and isinstance(sample.mask_path, str) and sample.mask_path != "":
            gt_mask = read_mask(sample.mask_path, as_tensor=True).float()
            if gt_mask.ndim == 2:
                gt_mask = gt_mask.unsqueeze(0)
            # 마스크도 이미지와 동일한 크기로 리사이즈
            gt_mask = torch.nn.functional.interpolate(
                gt_mask.unsqueeze(0), size=self.image_size, mode="nearest"
            ).squeeze(0)

        if self.preprocess is not None:
            image = self.preprocess(image)

        if gt_mask is None and isinstance(image, torch.Tensor):
            gt_mask = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32)

        return ImageItem(
            image_path=sample.image_path,
            image=image,
            gt_label=torch.tensor(sample.label_index, dtype=torch.long),
            gt_mask=gt_mask,
        )


class GoodsADDataset(AnomalibDataset):
    """GoodsAD 커스텀 Dataset (이미지 jpg, 마스크 png 처리)"""
    def __init__(
        self,
        root: Path | str,
        category: str,
        split: str = "train",
        preprocess=None,
        image_size: tuple[int, int] = (256, 256),
    ):
        super().__init__(augmentations=None)
        self.root = Path(root)
        self._category = category
        self.split = split
        self._samples = self.make_dataset()
        self.preprocess = preprocess
        self.image_size = image_size

    def make_dataset(self) -> pd.DataFrame:
        """GoodsAD 샘플 DataFrame 생성"""
        cat_path = self.root / self.category
        samples_list = []

        if self.split == "train":
            train_good = cat_path / "train" / "good"
            if train_good.exists():
                for img_path in sorted(train_good.glob("*.jpg")):
                    if img_path.name.startswith("."):
                        continue
                    samples_list.append({
                        "image_path": str(img_path),
                        "label": "normal",
                        "label_index": 0,
                        "mask_path": None,
                        "split": "train",
                    })
        else:
            test_dir = cat_path / "test"
            gt_dir = cat_path / "ground_truth"

            if test_dir.exists():
                for defect_dir in sorted(test_dir.iterdir()):
                    if not defect_dir.is_dir() or defect_dir.name.startswith("."):
                        continue

                    defect_type = defect_dir.name
                    is_normal = defect_type == "good"

                    for img_path in sorted(defect_dir.glob("*.jpg")):
                        if img_path.name.startswith("."):
                            continue

                        mask_path = None
                        if not is_normal:
                            # GoodsAD: 마스크는 .png, 이미지와 동일한 stem
                            mask_file = gt_dir / defect_type / f"{img_path.stem}.png"
                            if mask_file.exists():
                                mask_path = str(mask_file)

                        samples_list.append({
                            "image_path": str(img_path),
                            "label": "normal" if is_normal else "anomaly",
                            "label_index": 0 if is_normal else 1,
                            "mask_path": mask_path,
                            "split": "test",
                        })

        samples = pd.DataFrame(samples_list)
        samples.attrs["task"] = "segmentation"
        return samples

    def __getitem__(self, index: int) -> ImageItem:
        sample = self.samples.iloc[index]
        image_np = read_image(sample.image_path)

        # PIL.Image.fromarray가 float64 타입을 처리 못하므로 uint8로 변환
        if image_np.dtype in [np.float32, np.float64]:
            image_np = (image_np * 255).astype(np.uint8)
        image = Image.fromarray(image_np)

        gt_mask = None
        if pd.notna(sample.mask_path) and isinstance(sample.mask_path, str) and sample.mask_path != "":
            gt_mask = read_mask(sample.mask_path, as_tensor=True).float()
            if gt_mask.ndim == 2:
                gt_mask = gt_mask.unsqueeze(0)
            # 마스크도 이미지와 동일한 크기로 리사이즈
            gt_mask = torch.nn.functional.interpolate(
                gt_mask.unsqueeze(0), size=self.image_size, mode="nearest"
            ).squeeze(0)

        if self.preprocess is not None:
            image = self.preprocess(image)

        if gt_mask is None and isinstance(image, torch.Tensor):
            gt_mask = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32)

        return ImageItem(
            image_path=sample.image_path,
            image=image,
            gt_label=torch.tensor(sample.label_index, dtype=torch.long),
            gt_mask=gt_mask,
        )


class GoodsADDataModule(LightningDataModule):
    """GoodsAD 커스텀 DataModule"""
    def __init__(self, root, category, image_size: tuple[int, int] = (256, 256), train_batch_size=32, eval_batch_size=32, num_workers=6, **kwargs):
        super().__init__()
        self.root = Path(root)
        self.category = category
        self.image_size = image_size
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self._name = "GoodsAD"
        self.transform = Compose([Resize(image_size, antialias=True), ToTensor()])
        self.train_data = None
        self.test_data = None

    @property
    def name(self) -> str:
        return self._name

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_data = GoodsADDataset(self.root, self.category, split="train", preprocess=self.transform, image_size=self.image_size)
            self.test_data = GoodsADDataset(self.root, self.category, split="test", preprocess=self.transform, image_size=self.image_size)
        if stage in ("test", "predict"):
            if self.test_data is None:
                self.test_data = GoodsADDataset(self.root, self.category, split="test", preprocess=self.transform, image_size=self.image_size)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_items)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_items)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_items)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_items)


class MVTecLOCODataModule(LightningDataModule):
    """MVTec-LOCO 커스텀 DataModule"""
    def __init__(
        self,
        root: str | Path,
        category: str,
        image_size: tuple[int, int] = (256, 256),
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.root = Path(root)
        self.category = category
        self.image_size = image_size
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self._name = "MVTec-LOCO"
        self.transform = Compose([Resize(image_size, antialias=True), ToTensor()])

        self.train_data = None
        self.test_data = None

    @property
    def name(self) -> str:
        return self._name

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_data = MVTecLOCODataset(root=self.root, category=self.category, split="train", preprocess=self.transform, image_size=self.image_size)
            self.test_data = MVTecLOCODataset(root=self.root, category=self.category, split="test", preprocess=self.transform, image_size=self.image_size)
        if stage == "test" or stage == "predict":
            if self.test_data is None:
                self.test_data = MVTecLOCODataset(root=self.root, category=self.category, split="test", preprocess=self.transform, image_size=self.image_size)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_items)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_items)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_items)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_items)


class MMADLoader:
    def __init__(self, config: dict, model_name: str):
        self.config = config
        self.model_name = model_name
        self.root = Path(config["data"]["root"])
        self.datasets_to_run = config["data"].get("datasets", ["MVTec-LOCO"])
        self.EXCLUDE_DIRS = {"split_csv", "visa_pytorch"}

    def get_categories(self, dataset: str) -> list[str]:
        ds_path = self.root / dataset
        if not ds_path.exists():
            return []
        return sorted([
            d.name for d in ds_path.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name not in self.EXCLUDE_DIRS
        ])

    def mvtec_ad(self, category: str, **kwargs) -> AnomalibDataModule:
        return MVTecAD(root=str(self.root / "MVTec-AD"), category=category, **kwargs)

    def visa(self, category: str, **kwargs) -> AnomalibDataModule:
        return Visa(root=str(self.root / "VisA"), category=category, **kwargs)

    def mvtec_loco(self, category: str, **kwargs) -> MVTecLOCODataModule:
        return MVTecLOCODataModule(root=str(self.root / "MVTec-LOCO"), category=category, **kwargs)

    def goods_ad(self, category: str, **kwargs) -> GoodsADDataModule:
        return GoodsADDataModule(root=str(self.root / "GoodsAD"), category=category, **kwargs)

    def get_datamodule(self, dataset: str, category: str, **kwargs):
        """DataModule 생성. config에서 기본값 로드 후, kwargs로 오버라이드 가능."""
        img_size = tuple(self.config.get("data", {}).get("image_size", (256, 256)))
        is_eff = (self.model_name == "efficientad")
        training_cfg = self.config.get("training", {})

        # config 기반 기본값
        default_kwargs = {
            "image_size": img_size,
            "num_workers": training_cfg.get("num_workers", 4),
            "train_batch_size": 1 if is_eff else training_cfg.get("train_batch_size", 32),
            "eval_batch_size": training_cfg.get("eval_batch_size", 32),
        }
        # 전달받은 kwargs로 오버라이드
        default_kwargs.update(kwargs)

        loaders = {
            "MVTec-AD": self.mvtec_ad,
            "VisA": self.visa,
            "MVTec-LOCO": self.mvtec_loco,
            "GoodsAD": self.goods_ad,
        }

        if dataset not in loaders:
            raise ValueError(f"Unknown dataset: {dataset}. Available: {list(loaders.keys())}")
        return loaders[dataset](category, **default_kwargs)

    def iter_all(self, **kwargs) -> Generator[tuple[str, str, LightningDataModule], None, None]:
        for dataset in self.datasets_to_run:
            categories = self.get_categories(dataset)
            for category in categories:
                datamodule = self.get_datamodule(dataset, category, **kwargs)
                yield dataset, category, datamodule
