import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from .path import get_logs_dir


def setup_logger(
    name: str = "mmad_inspector",
    log_dir: Optional[str] = None,
    log_level: str = "INFO",
    log_prefix: Optional[str] = None,
    file_logging: bool = True,
) -> logging.Logger:
    """
    Args:
        name: 로거 이름
        log_dir: 로그 저장 디렉토리 (None이면 프로젝트 루트/logs)
        log_level: 로그 레벨
        log_prefix: 로그 파일 prefix (None이면 파일 로깅 비활성화)
        file_logging: 파일 로깅 활성화 여부
    """
    # 로거 설정
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # 핸들러 중복 추가 방지
    if logger.hasHandlers():
        logger.handlers.clear()

    # 파일 로깅 (log_prefix가 있을 때만)
    if file_logging and log_prefix is not None:
        # 로그 디렉토리 설정
        if log_dir is None:
            log_dir = get_logs_dir()
        else:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

        # 로그 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{log_prefix}_{timestamp}.log"

        # 파일 핸들러
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # 콘솔 핸들러 (항상 추가)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger
