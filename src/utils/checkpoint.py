from __future__ import annotations
from pathlib import Path
import torch
from .log import setup_logger

# log_prefix=None으로 파일 로깅 비활성화 (콘솔만 출력)
logger = setup_logger(name="Checkpoint", log_prefix=None)

def save_checkpoint(model: torch.nn.Module, path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, path)
    logger.info(f"Saved: {path}")
    return path

def load_checkpoint(model: torch.nn.Module, path: str, device,) -> torch.nn.Module:
    if not Path(path).exists():
        logger.error(f"Checkpoint not found: {path}")
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    logger.info(f"Loaded: {path}")
    return model