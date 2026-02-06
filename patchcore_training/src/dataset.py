"""Dataset module for anomaly detection.

Supports:
- MVTec-LOCO: ground_truth/{defect_type}/{image_name}_mask.png
- GoodsAD: ground_truth/{defect_type}/{image_name}.png
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class AnomalyDataset(Dataset):
    """Anomaly detection dataset for MVTec-LOCO and GoodsAD.

    Args:
        root: Dataset root directory
        dataset_name: Dataset name (MVTec-LOCO, GoodsAD)
        category: Category/class name
        split: 'train' or 'test'
        image_size: Target image size
        include_mask: Whether to load ground truth masks (test only)
    """

    # ImageNet normalization
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        root: Union[str, Path],
        dataset_name: str,
        category: str,
        split: str = "train",
        image_size: int = 224,
        include_mask: bool = True,
    ):
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.category = category
        self.split = split
        self.image_size = image_size
        self.include_mask = include_mask and (split == "test")

        # Build transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        # Load file list
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict]:
        """Load sample file paths and labels."""
        samples = []
        category_dir = self.root / self.dataset_name / self.category

        if self.split == "train":
            # Training: only good samples
            train_dir = category_dir / "train" / "good"
            if train_dir.exists():
                for img_path in sorted(train_dir.glob("*")):
                    # Skip macOS hidden files
                    if img_path.name.startswith("._"):
                        continue
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                        samples.append({
                            "image_path": img_path,
                            "label": 0,  # normal
                            "mask_path": None,
                            "defect_type": "good",
                        })
        else:
            # Test: good + anomaly samples
            test_dir = category_dir / "test"
            if test_dir.exists():
                for defect_dir in sorted(test_dir.iterdir()):
                    if not defect_dir.is_dir():
                        continue

                    defect_type = defect_dir.name
                    is_normal = defect_type == "good"

                    for img_path in sorted(defect_dir.glob("*")):
                        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                            continue
                        # Skip macOS hidden files
                        if img_path.name.startswith("._"):
                            continue

                        mask_path = None
                        if not is_normal and self.include_mask:
                            mask_path = self._get_mask_path(img_path, defect_type)

                        samples.append({
                            "image_path": img_path,
                            "label": 0 if is_normal else 1,
                            "mask_path": mask_path,
                            "defect_type": defect_type,
                        })

        return samples

    def _get_mask_path(self, img_path: Path, defect_type: str) -> Optional[Path]:
        """Get mask path based on dataset format.

        MVTec-LOCO: ground_truth/{defect_type}/{stem}/000.png (nested folder, always 000.png inside)
        GoodsAD: ground_truth/{defect_type}/{stem}.png (flat)
        """
        gt_dir = img_path.parent.parent.parent / "ground_truth" / defect_type

        if self.dataset_name == "MVTec-LOCO":
            # MVTec-LOCO: {stem}/000.png (nested folder, mask is always named 000.png)
            mask_path = gt_dir / img_path.stem / "000.png"
            if mask_path.exists():
                return mask_path
            # Fallback 1: try {stem}/{stem}.png
            mask_path = gt_dir / img_path.stem / f"{img_path.stem}.png"
            if mask_path.exists():
                return mask_path
            # Fallback 2: try flat structure
            mask_path = gt_dir / f"{img_path.stem}.png"
            if mask_path.exists():
                return mask_path
        else:
            # GoodsAD and others: {stem}.png (flat structure)
            mask_path = gt_dir / f"{img_path.stem}.png"
            if mask_path.exists():
                return mask_path

        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load image (BGR -> RGB)
        image = cv2.imread(str(sample["image_path"]))
        if image is None:
            raise ValueError(f"Failed to load image: {sample['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        image_tensor = self.transform(image)

        result = {
            "image": image_tensor,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "image_path": str(sample["image_path"]),
            "defect_type": sample["defect_type"],
        }

        # Load mask if available
        if self.include_mask and sample["mask_path"] is not None:
            mask = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = (mask > 0).astype(np.uint8) * 255
                mask_tensor = self.mask_transform(mask)
                result["mask"] = mask_tensor
            else:
                result["mask"] = torch.zeros(1, self.image_size, self.image_size)
        else:
            result["mask"] = torch.zeros(1, self.image_size, self.image_size)

        return result


class InferenceDataset(Dataset):
    """Dataset for inference from mmad.json format.

    Args:
        data_root: Root directory for images
        image_paths: List of relative image paths
        image_size: Target image size
    """

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        data_root: Union[str, Path],
        image_paths: List[str],
        image_size: int = 224,
    ):
        self.data_root = Path(data_root)
        self.image_paths = image_paths
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        rel_path = self.image_paths[idx]
        full_path = self.data_root / rel_path

        # Parse dataset and category from path
        parts = rel_path.split("/")
        dataset_name = parts[0] if len(parts) > 0 else ""
        category = parts[1] if len(parts) > 1 else ""

        # Load image
        image = cv2.imread(str(full_path))
        if image is None:
            return {
                "image": torch.zeros(3, self.image_size, self.image_size),
                "image_path": rel_path,
                "dataset": dataset_name,
                "category": category,
                "valid": False,
                "original_size": (0, 0),
            }

        original_size = image.shape[:2]  # (H, W)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image)

        return {
            "image": image_tensor,
            "image_path": rel_path,
            "dataset": dataset_name,
            "category": category,
            "valid": True,
            "original_size": original_size,
        }


def get_dataloader(
    root: Union[str, Path],
    dataset_name: str,
    category: str,
    split: str = "train",
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    include_mask: bool = True,
) -> DataLoader:
    """Create dataloader for anomaly detection.

    Args:
        root: Dataset root
        dataset_name: Dataset name
        category: Category name
        split: 'train' or 'test'
        image_size: Image size
        batch_size: Batch size
        num_workers: Number of workers
        include_mask: Include masks (test only)

    Returns:
        DataLoader instance
    """
    dataset = AnomalyDataset(
        root=root,
        dataset_name=dataset_name,
        category=category,
        split=split,
        image_size=image_size,
        include_mask=include_mask,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
