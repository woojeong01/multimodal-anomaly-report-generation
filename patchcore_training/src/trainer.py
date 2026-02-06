"""PatchCore trainer module."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

from .dataset import get_dataloader
from .model import PatchCore


class PatchCoreTrainer:
    """Trainer for PatchCore models.

    Handles training (memory bank construction) for multiple datasets/categories.
    """

    def __init__(
        self,
        config: Dict,
        device: torch.device = None,
    ):
        """Initialize trainer.

        Args:
            config: Configuration dictionary
            device: Device to use
        """
        self.config = config

        if device is None:
            device_str = config.get("device", "cuda")
            if device_str == "cuda" and torch.cuda.is_available():
                device = torch.device("cuda")
            elif device_str == "mps" and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
        print(f"Using device: {self.device}")

        # Data settings
        self.data_root = Path(config["data"]["root"])
        self.image_size = config["data"].get("image_size", 224)

        # Model settings
        model_config = config.get("model", {})
        self.backbone = model_config.get("backbone", "wide_resnet50_2")
        self.layers = model_config.get("layers", ["layer2", "layer3"])
        self.coreset_ratio = model_config.get("coreset_ratio", 0.01)
        self.n_neighbors = model_config.get("n_neighbors", 9)

        # Training settings
        training_config = config.get("training", {})
        self.batch_size = training_config.get("batch_size", 32)
        self.num_workers = training_config.get("num_workers", 4)

        # Output settings
        output_config = config.get("output", {})
        self.checkpoint_dir = Path(output_config.get("checkpoint_dir", "checkpoints"))
        self.save_pt = output_config.get("save_pt", True)
        self.save_onnx = output_config.get("save_onnx", True)

    def get_dataset_categories(self) -> List[Tuple[str, str]]:
        """Get list of (dataset, category) tuples from config."""
        categories = []
        datasets_config = self.config["data"].get("datasets", {})

        for dataset_name, category_list in datasets_config.items():
            for category in category_list:
                categories.append((dataset_name, category))

        return categories

    def train_category(
        self,
        dataset_name: str,
        category: str,
        verbose: bool = True,
    ) -> PatchCore:
        """Train PatchCore for a single category.

        Args:
            dataset_name: Dataset name
            category: Category name
            verbose: Print progress

        Returns:
            Trained PatchCore model
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training: {dataset_name}/{category}")
            print(f"{'='*60}")

        # Create model
        model = PatchCore(
            backbone=self.backbone,
            layers=self.layers,
            coreset_ratio=self.coreset_ratio,
            n_neighbors=self.n_neighbors,
        )

        # Create dataloader
        dataloader = get_dataloader(
            root=self.data_root,
            dataset_name=dataset_name,
            category=category,
            split="train",
            image_size=self.image_size,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        if len(dataloader.dataset) == 0:
            print(f"Warning: No training samples found for {dataset_name}/{category}")
            return None

        if verbose:
            print(f"Training samples: {len(dataloader.dataset)}")

        # Fit model (build memory bank)
        model.fit(dataloader, self.device)

        # Save model
        saved_paths = model.save(
            save_dir=self.checkpoint_dir,
            dataset_name=dataset_name,
            category=category,
            save_pt=self.save_pt,
            save_onnx=self.save_onnx,
            image_size=self.image_size,
        )

        if verbose:
            print(f"Model saved to: {saved_paths}")

        return model

    def train_all(self, verbose: bool = True) -> Dict[str, PatchCore]:
        """Train PatchCore for all categories in config.

        Args:
            verbose: Print progress

        Returns:
            Dictionary mapping "dataset/category" to trained models
        """
        categories = self.get_dataset_categories()
        total = len(categories)

        print(f"\nTraining {total} categories")
        print(f"Checkpoint dir: {self.checkpoint_dir}")

        models = {}
        pbar = tqdm(categories, desc="Training categories")
        for dataset_name, category in pbar:
            pbar.set_description(f"Training {dataset_name}/{category}")

            model = self.train_category(dataset_name, category, verbose=verbose)

            if model is not None:
                key = f"{dataset_name}/{category}"
                models[key] = model

        print(f"\n{'='*60}")
        print(f"Training complete: {len(models)}/{total} categories")
        print(f"{'='*60}")

        return models

    def load_model(self, dataset_name: str, category: str) -> Optional[PatchCore]:
        """Load trained model for a category.

        Args:
            dataset_name: Dataset name
            category: Category name

        Returns:
            Loaded PatchCore model or None if not found
        """
        pt_path = self.checkpoint_dir / dataset_name / category / "model.pt"

        if not pt_path.exists():
            return None

        return PatchCore.load(str(pt_path), self.device)

    def load_all_models(self) -> Dict[str, PatchCore]:
        """Load all available trained models.

        Returns:
            Dictionary mapping "dataset/category" to models
        """
        models = {}
        categories = self.get_dataset_categories()

        for dataset_name, category in tqdm(categories, desc="Loading models"):
            model = self.load_model(dataset_name, category)
            if model is not None:
                key = f"{dataset_name}/{category}"
                models[key] = model

        print(f"Loaded {len(models)} models")
        return models
