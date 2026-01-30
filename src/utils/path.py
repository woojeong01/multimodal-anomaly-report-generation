"""프로젝트 경로 유틸리티

Local / Colab 환경 모두 지원
"""
from __future__ import annotations
from pathlib import Path


def get_project_root() -> Path:
    """프로젝트 루트 경로 반환 (Local / Colab 모두 지원)

    Returns:
        프로젝트 루트 경로 (multiModal_anomaly_report/)

    Example:
        >>> from src.utils.path import get_project_root
        >>> root = get_project_root()
        >>> data_path = root / "dataset" / "MMAD"
    """
    # Colab 환경 체크
    try:
        import google.colab
        colab_paths = [
            Path('/content/drive/Othercomputers/내 Mac/multiModal_anomaly_report'),
            Path('/content/drive/MyDrive/multiModal_anomaly_report'),
        ]
        for p in colab_paths:
            if p.exists():
                return p
        raise FileNotFoundError("Colab에서 프로젝트 경로를 찾을 수 없습니다.")
    except ImportError:
        pass

    # Local 환경 - src/utils/path.py 기준으로 상위 2단계
    current = Path(__file__).resolve()
    project_root = current.parents[2]  # src/utils -> src -> project_root

    # 검증: src 폴더가 존재하는지 확인
    if (project_root / "src").exists():
        return project_root

    raise FileNotFoundError("프로젝트 루트를 찾을 수 없습니다.")


def get_logs_dir() -> Path:
    """로그 디렉토리 경로 반환 (없으면 생성)"""
    logs_dir = get_project_root() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def get_checkpoints_dir() -> Path:
    """체크포인트 디렉토리 경로 반환 (없으면 생성)"""
    ckpt_dir = get_project_root() / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def get_outputs_dir() -> Path:
    """출력 디렉토리 경로 반환 (없으면 생성)"""
    outputs_dir = get_project_root() / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir
