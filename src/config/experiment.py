"""Experiment configuration loader."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from src.service.settings import load_yaml


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    ad_model: Optional[str] = None       # patchcore | winclip | efficientad | null
    llm: str = "qwen"                    # model name from MODEL_REGISTRY
    few_shot: int = 1
    similar_template: bool = True
    max_images: Optional[int] = None
    data_root: Optional[str] = None
    mmad_json: Optional[str] = None
    max_image_size: Tuple[int, int] = (512, 512)  # LLM에 전달할 이미지 최대 크기
    output_dir: str = "outputs/eval"
    batch_mode: bool = False
    resume: bool = False

    @property
    def experiment_name(self) -> str:
        """Auto-generate experiment name: {ad_model}_{llm}_{few_shot}shot"""
        ad = self.ad_model or "no_ad"
        return f"{ad}_{self.llm}_{self.few_shot}shot"


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load ExperimentConfig from a YAML file.

    Supports ``${ENV_VAR:-default}`` expansion via ``load_yaml()``.
    """
    d = load_yaml(path)

    eval_section = d.get("eval", {})

    return ExperimentConfig(
        ad_model=d.get("ad_model"),
        llm=d.get("llm", "qwen"),
        few_shot=eval_section.get("few_shot", 1),
        similar_template=eval_section.get("similar_template", True),
        max_images=eval_section.get("max_images"),
        max_image_size=tuple(eval_section.get("max_image_size", [512, 512])),
        data_root=d.get("data_root"),
        mmad_json=d.get("mmad_json"),
        output_dir=d.get("output_dir", "outputs/eval"),
        batch_mode=eval_section.get("batch_mode", False),
        resume=eval_section.get("resume", False),
    )
