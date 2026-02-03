from pathlib import Path
from typing import Generator

from anomalib.data import MVTecAD, Visa, MVTecLOCO, Folder
from anomalib.data.datamodules.base import AnomalibDataModule


class MMADLoader:
    # DATASETS = ["GoodsAD"] # 단일 Test
    DATASETS = ["MVTec-AD", "VisA", "GoodsAD", "MVTec-LOCO"]

    # 카테고리가 아닌 폴더 제외
    EXCLUDE_DIRS = {"split_csv", "visa_pytorch"}

    def __init__(self, root: str = "dataset/MMAD"):
        self.root = Path(root)

    def get_categories(self, dataset: str) -> list[str]:
        ds_path = self.root / dataset
        if not ds_path.exists():
            return []
        return sorted([
            d.name for d in ds_path.iterdir()
            if d.is_dir()
            and not d.name.startswith(".")
            and d.name not in self.EXCLUDE_DIRS
        ])

    def mvtec_ad(self, category: str, **kwargs) -> AnomalibDataModule:
        return MVTecAD(
            root=str(self.root / "MVTec-AD"),
            category=category,
            **kwargs
        )

    def visa(self, category: str, **kwargs) -> AnomalibDataModule:
        return Visa(
            root=str(self.root / "VisA"),
            category=category,
            **kwargs
        )

    # def visa(self, category: str, **kwargs) -> AnomalibDataModule:
    #     return Folder(
    #         name=category,
    #         root=str(self.root / "VisA" / "visa_pytorch" / category),
    #         normal_dir="train/good",
    #         abnormal_dir="test/bad",
    #         normal_test_dir="test/good",
    #         mask_dir="ground_truth",
    #         **kwargs
    #     )

    def mvtec_loco(self, category: str, **kwargs) -> AnomalibDataModule:
        return MVTecLOCO(
            root=str(self.root / "MVTec-LOCO"),
            category=category,
            **kwargs
        )

    def goods_ad(self, category: str, **kwargs) -> AnomalibDataModule:
        cat_path = self.root / "GoodsAD" / category
        return Folder(
            name=category,
            root=str(cat_path),
            normal_dir="train/good",
            abnormal_dir="test",
            normal_test_dir="test/good",
            # mask_dir 제외 - 학습에는 불필요, 이미지/마스크 확장자 충돌 방지
            extensions=[".jpg"],  # .txt 파일 제외
            **kwargs
        )

    def get_datamodule(self, dataset: str, category: str, **kwargs) -> AnomalibDataModule:
        loaders = {
            "MVTec-AD": self.mvtec_ad,
            "VisA": self.visa,
            "MVTec-LOCO": self.mvtec_loco,
            "GoodsAD": self.goods_ad,
        }

        if dataset not in loaders:
            raise ValueError(f"Unknown dataset: {dataset}. Available: {list(loaders.keys())}")
        return loaders[dataset](category, **kwargs)

    def iter_all(self, **kwargs) -> Generator[tuple[str, str, AnomalibDataModule], None, None]:
        for dataset in self.DATASETS:
            categories = self.get_categories(dataset)
            for category in categories:
                datamodule = self.get_datamodule(dataset, category, **kwargs)
                yield dataset, category, datamodule
