from __future__ import annotations

import platform
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import rc
import seaborn as sns
from PIL import Image
import torch


def set_korean_font(verbose: bool = False) -> bool:
    """한글 폰트 설정. Mac, Windows, Linux 지원. verbose=True면 메시지 출력."""
    system = platform.system()
    success = False
    font_name = None

    # OS별 한글 폰트 후보 목록
    font_candidates = {
        'Darwin': ['Arial Unicode MS', 'AppleGothic', 'Apple SD Gothic Neo'],
        'Windows': ['Malgun Gothic', 'NanumGothic', 'Gulim', 'Dotum'],
        'Linux': ['Noto Sans CJK KR', 'NanumGothic', 'NanumBarunGothic', 'UnDotum', 'DejaVu Sans']
    }

    # 시스템에 설치된 폰트 목록
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}

    # 해당 OS의 폰트 후보에서 사용 가능한 폰트 찾기
    candidates = font_candidates.get(system, font_candidates['Linux'])

    for font in candidates:
        if font in available_fonts:
            font_name = font
            break

    if font_name:
        rc('font', family=font_name)
        success = True
        if verbose:
            print(f'Korean font set: {font_name} ({system})')
    else:
        if verbose:
            print(f'No Korean font found for {system}. Tried: {candidates}')
        success = False

    plt.rcParams['axes.unicode_minus'] = False
    return success


def count_plot(df, col, ax=None, figsize=(10, 6), palette="Blues_r", rotation=None, title=None, xlabel=None, ylabel=None, order='desc', orient='v', top_n=None, show=True):
    """
    order: 'desc'(내림차순), 'asc'(오름차순), None(정렬 안 함)
    orient: 'v'(세로, 기본값), 'h'(가로)
    top_n: 상위 N개만 표시 (None이면 전체)
    """
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    # 정렬 순서 결정
    if order == 'desc':
        order_list = df[col].value_counts().index.tolist()
    elif order == 'asc':
        order_list = df[col].value_counts(ascending=True).index.tolist()
    else:
        order_list = df[col].value_counts().index.tolist()

    # top_n 적용
    if top_n is not None:
        order_list = order_list[:top_n]
        df = df[df[col].isin(order_list)]

    # 방향에 따라 x/y 설정
    if orient == 'h':
        sns.countplot(data=df, y=col, palette=palette, ax=ax, order=order_list)
        ax.set_ylabel(ylabel if ylabel is not None else col)
        ax.set_xlabel(xlabel if xlabel is not None else 'Count')
    else:
        sns.countplot(data=df, x=col, palette=palette, ax=ax, order=order_list)
        ax.set_xlabel(xlabel if xlabel is not None else col)
        ax.set_ylabel(ylabel if ylabel is not None else 'Count')

    ax.set_title(title if title else f'{col} Count')
    if rotation:
        ax.tick_params(axis='x' if orient == 'v' else 'y', rotation=rotation)

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def bar_plot(df, x_col, y_col, ax=None, figsize=(10, 6), hue=None, palette="Blues_r", rotation=None, title=None, xlabel=None, ylabel=None, show=True):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    if hue:
        sns.barplot(x=x_col, y=y_col, hue=hue, data=df, palette=palette, ax=ax)
    else:
        sns.barplot(x=x_col, y=y_col, data=df, palette=palette, ax=ax)

    if rotation:
        ax.tick_params(axis='x', rotation=rotation)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def line_plot(df, x_col, y_col, ax=None, figsize=(12, 6), color=None, marker='o', linewidth=2, rotation=None, title=None, show=True):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    sns.lineplot(x=df[x_col], y=df[y_col], data=df, color=color, marker=marker, linewidth=linewidth, ax=ax)
    if rotation:
        ax.tick_params(axis='x', rotation=rotation)
    ax.set_title(title if title else f'{y_col} by {x_col} (Line Plot)')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True)

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def box_plot(df, col, hue=None, ax=None, figsize=(8, 6), palette=None, title=None, show=True):
    """
    단일 boxplot 또는 hue로 그룹 비교
    - hue: 그룹 비교할 컬럼명 (예: 'completed')
    - palette: 색상 리스트 (예: ['salmon', 'skyblue'])
    """
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    if palette is None:
        palette = ['skyblue', 'salmon']

    if hue:
        sns.boxplot(data=df, x=hue, y=col, palette=palette, ax=ax)
        ax.set_title(title if title else f'{col} by {hue}')
    else:
        sns.boxplot(y=df[col], color=palette[0], ax=ax)
        ax.set_title(title if title else f'{col} (Box Plot)')

    ax.set_ylabel(col)

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def hist_plot(df, col, ax=None, figsize=(10, 6), bins='auto', color='steelblue', kde=False, stat='count', title=None, xlabel=None, ylabel=None, show=True):
    """
    히스토그램 시각화
    - bins: 구간 수 ('auto', 정수, 리스트 등)
    - kde: True면 KDE 곡선 함께 표시
    - stat: 'count', 'density', 'probability', 'frequency' 등
    """
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    sns.histplot(data=df, x=col, bins=bins, color=color, kde=kde, stat=stat, ax=ax)

    ax.set_title(title if title else f'{col} Distribution')
    ax.set_xlabel(xlabel if xlabel else col)
    ax.set_ylabel(ylabel if ylabel else stat.capitalize())

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def kde_plot(df, col, ax=None, figsize=(10, 6), color='steelblue', fill=True, alpha=0.3, linewidth=2, hue=None, palette='Blues', title=None, xlabel=None, ylabel=None, show=True):
    """
    KDE(커널 밀도 추정) 시각화
    - fill: True면 곡선 아래 영역 채움
    - alpha: 채움 투명도 (0~1)
    - hue: 그룹별 비교할 컬럼명
    - palette: hue 사용 시 색상 팔레트
    """
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    if hue:
        sns.kdeplot(data=df, x=col, hue=hue, fill=fill, alpha=alpha, linewidth=linewidth, palette=palette, ax=ax)
    else:
        sns.kdeplot(data=df, x=col, color=color, fill=fill, alpha=alpha, linewidth=linewidth, ax=ax)

    ax.set_title(title if title else f'{col} KDE')
    ax.set_xlabel(xlabel if xlabel else col)
    ax.set_ylabel(ylabel if ylabel else 'Density')

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def heatmap_plot(data, ax=None, figsize=(12, 8), cmap='Blues', annot=True, fmt='d', linewidths=0.5, cbar=True, title=None, xlabel=None, ylabel=None, rotation_x=45, rotation_y=0, show=True):
    """
    히트맵 시각화 (crosstab, pivot table, correlation matrix 등에 사용)
    - data: 2D 데이터 (DataFrame, crosstab 결과 등)
    - cmap: 색상맵 ('Blues', 'Reds', 'YlGnBu', 'coolwarm' 등)
    - annot: True면 셀에 값 표시
    - fmt: annot 포맷 ('d'=정수, '.2f'=소수점 2자리 등)
    - linewidths: 셀 간 선 두께
    - cbar: True면 컬러바 표시
    - rotation_x, rotation_y: x축, y축 레이블 회전 각도
    """
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    sns.heatmap(data, annot=annot, fmt=fmt, cmap=cmap, linewidths=linewidths, cbar=cbar, ax=ax)

    ax.set_title(title if title else 'Heatmap')
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if rotation_x:
        ax.tick_params(axis='x', rotation=rotation_x)
    if rotation_y:
        ax.tick_params(axis='y', rotation=rotation_y)

    if show:
        plt.tight_layout()
        plt.show()
    return ax


# Anomaly Detection Visualization Functions

def tensor_to_numpy(tensor):
    """Tensor를 numpy 배열로 변환"""
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3:
            if tensor.shape[0] in [1, 3]:
                tensor = tensor.permute(1, 2, 0)
            if tensor.shape[-1] == 1:
                tensor = tensor.squeeze(-1)
        tensor = tensor.numpy()
    return tensor


def load_original_image(image_path: str | Path, target_size: tuple = None) -> np.ndarray:
    """원본 이미지를 직접 로드 (정규화 없이)"""
    img = Image.open(image_path).convert("RGB")
    if target_size:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(img) / 255.0  # 0-1 스케일


def visualize_anomaly_prediction(
    image_path: str | Path,
    anomaly_map: np.ndarray | torch.Tensor = None,
    pred_mask: np.ndarray | torch.Tensor = None,
    gt_mask: np.ndarray | torch.Tensor = None,
    pred_score: float = None,
    pred_label: int = None,
    figsize: tuple = (16, 4),
    cmap: str = "jet",
    alpha: float = 0.5,
    title: str = None,
    show: bool = True,
) -> plt.Figure:
    """
    Anomaly Detection 예측 결과 시각화

    Args:
        image_path: 원본 이미지 경로 (검은 이미지 문제 해결을 위해 직접 로드)
        anomaly_map: 이상 히트맵
        pred_mask: 예측 마스크
        gt_mask: Ground Truth 마스크
        pred_score: 예측 점수
        pred_label: 예측 라벨 (0: normal, 1: anomaly)
        figsize: figure 크기
        cmap: anomaly_map 컬러맵
        alpha: overlay 투명도
        title: 전체 제목
        show: plt.show() 호출 여부

    Returns:
        matplotlib Figure
    """
    # 원본 이미지 직접 로드 (정규화 문제 해결)
    image = load_original_image(image_path)

    # Tensor → Numpy 변환
    anomaly_map = tensor_to_numpy(anomaly_map)
    pred_mask = tensor_to_numpy(pred_mask)
    gt_mask = tensor_to_numpy(gt_mask)

    # anomaly_map 크기 맞추기 & 정규화
    if anomaly_map is not None:
        if anomaly_map.shape[:2] != image.shape[:2]:
            from skimage.transform import resize
            anomaly_map = resize(anomaly_map, image.shape[:2], preserve_range=True)
        if anomaly_map.max() > anomaly_map.min():
            anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())

    # pred_mask 크기 맞추기
    if pred_mask is not None and pred_mask.shape[:2] != image.shape[:2]:
        from skimage.transform import resize
        pred_mask = resize(pred_mask, image.shape[:2], preserve_range=True)

    # gt_mask 크기 맞추기
    if gt_mask is not None and gt_mask.shape[:2] != image.shape[:2]:
        from skimage.transform import resize
        gt_mask = resize(gt_mask, image.shape[:2], preserve_range=True)

    # 패널 구성
    panels = ["Image"]
    if gt_mask is not None:
        panels.append("GT Mask")
    if anomaly_map is not None:
        panels.append("Image + Anomaly Map")
    if pred_mask is not None:
        panels.append("Image + Pred Mask")

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    # 1. Original Image
    axes[panel_idx].imshow(image)
    axes[panel_idx].set_title("Image", fontsize=12, color="blue")
    axes[panel_idx].axis("off")
    panel_idx += 1

    # 2. GT Mask
    if gt_mask is not None:
        axes[panel_idx].imshow(gt_mask, cmap="gray")
        axes[panel_idx].set_title("GT Mask", fontsize=12, color="blue")
        axes[panel_idx].axis("off")
        panel_idx += 1

    # 3. Image + Anomaly Map
    if anomaly_map is not None:
        axes[panel_idx].imshow(image)
        axes[panel_idx].imshow(anomaly_map, cmap=cmap, alpha=alpha)
        axes[panel_idx].set_title("Image + Anomaly Map", fontsize=12, color="blue")
        axes[panel_idx].axis("off")
        panel_idx += 1

    # 4. Image + Pred Mask (Contour)
    if pred_mask is not None:
        axes[panel_idx].imshow(image)
        if pred_mask.max() > 0:
            axes[panel_idx].contour(pred_mask, levels=[0.5], colors=["red"], linewidths=2)
        axes[panel_idx].set_title("Image + Pred Mask", fontsize=12, color="blue")
        axes[panel_idx].axis("off")

    # 제목
    if title:
        fig.suptitle(title, fontsize=14)

    # 점수 표시
    if pred_score is not None:
        label_text = "Anomaly" if (pred_label == 1 if pred_label is not None else pred_score > 0.5) else "Normal"
        fig.text(0.5, 0.02, f"Score: {float(pred_score):.4f} | Prediction: {label_text}",
                 ha="center", fontsize=11, color="green")

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def visualize_predictions_from_runner(
    runner,
    n_samples_per_category: int = 1,
    filter_by: str = "all",
    random_sample: bool = True,
    categories: list = None,
    show_inference_time: bool = True,
    figsize: tuple = (16, 4),
    show: bool = True,
) -> dict:
    """
    학습된 모델로 각 카테고리별 예측 및 시각화 (선택된 샘플만 inference)

    Args:
        runner: Anomalibs 인스턴스 (scripts/train_anomalib.py)
        n_samples_per_category: 카테고리당 시각화할 샘플 수
        filter_by: 필터링 옵션 ("all", "anomaly", "normal")
        random_sample: True면 랜덤 샘플링, False면 순서대로
        categories: 시각화할 카테고리 [("GoodsAD", "cigarette_box"), ...], None이면 전체
        show_inference_time: inference 시간 표시 여부
        figsize: 샘플당 figure 크기
        show: plt.show() 호출 여부

    Returns:
        dict: {(dataset, category): {"inference_time_ms": float, ...}}

    Example:
        >>> runner = Anomalibs()
        >>> # Defect 샘플만 랜덤으로 카테고리당 2개씩
        >>> results = visualize_predictions_from_runner(runner, n_samples_per_category=2, filter_by="anomaly")
    """
    import time
    import random
    from torch.utils.data import DataLoader, Subset
    from src.datasets.dataloader import collate_items

    results = {}

    # 카테고리 목록
    target_categories = categories if categories else runner.get_trained_categories()

    if not target_categories:
        print("No trained categories found.")
        return results

    print(f"Categories: {len(target_categories)} | Filter: {filter_by} | Samples: {n_samples_per_category}\n")

    for dataset, category in target_categories:
        print(f"{dataset} / {category}")

        # 1. DataModule 로드
        dm_kwargs = runner.get_datamodule_kwargs()
        datamodule = runner.loader.get_datamodule(dataset, category, **dm_kwargs)
        datamodule.setup(stage="predict")
        test_dataset = datamodule.test_data

        # 2. filter_by로 인덱스 필터링
        if filter_by == "anomaly":
            indices = [i for i in range(len(test_dataset)) if test_dataset.samples.iloc[i].label_index == 1]
        elif filter_by == "normal":
            indices = [i for i in range(len(test_dataset)) if test_dataset.samples.iloc[i].label_index == 0]
        else:
            indices = list(range(len(test_dataset)))

        if not indices:
            print(f"  No {filter_by} samples. Skipping.\n")
            continue

        # 3. 랜덤 샘플링
        n_select = min(n_samples_per_category, len(indices))
        if random_sample:
            selected_indices = random.sample(indices, n_select)
        else:
            selected_indices = indices[:n_select]

        print(f"  {filter_by}: {len(indices)} images → Selected: {n_select}")

        # 4. Subset DataLoader 생성
        subset = Subset(test_dataset, selected_indices)
        subset_loader = DataLoader(subset, batch_size=n_select, shuffle=False, num_workers=0, collate_fn=collate_items)

        # 5. 모델 로드
        model = runner.get_model()
        ckpt_path = runner.get_ckpt_path(dataset, category)

        if ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location=runner.device, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"], strict=False)

        model.eval()
        model.to(runner.device)

        # 6. 선택된 샘플만 inference
        for batch in subset_loader:
            images = batch.image.to(runner.device)

            start_time = time.time()
            with torch.no_grad():
                outputs = model(images)
            inference_time = time.time() - start_time

            # InferenceBatch는 객체이므로 attribute로 접근
            anomaly_maps = getattr(outputs, "anomaly_map", None)
            pred_scores = getattr(outputs, "pred_score", None)

            avg_time_ms = (inference_time / n_select) * 1000
            if show_inference_time:
                print(f"  Inference: {avg_time_ms:.1f}ms/image")

            results[(dataset, category)] = {
                "n_samples": n_select,
                "avg_inference_time_ms": avg_time_ms,
            }

            # 7. 시각화
            for i in range(len(batch.image_path)):
                gt_label_str = "Anomaly" if batch.gt_label[i] == 1 else "Normal"
                anomaly_map = anomaly_maps[i] if anomaly_maps is not None else None
                pred_score = float(pred_scores[i]) if pred_scores is not None else None
                pred_mask = (anomaly_map > 0.5).float() if anomaly_map is not None else None

                visualize_anomaly_prediction(
                    image_path=batch.image_path[i],
                    anomaly_map=anomaly_map,
                    pred_mask=pred_mask,
                    gt_mask=batch.gt_mask[i] if batch.gt_mask is not None else None,
                    pred_score=pred_score,
                    figsize=figsize,
                    title=f"{dataset} / {category} [GT: {gt_label_str}]",
                    show=show,
                )

        del model
        runner.cleanup_memory()
        print()

    return results


def visualize_single_prediction(
    model_name: str = "patchcore",
    dataset: str = None,
    category: str = None,
    sample_idx: int = 0,
    config_path: str = "configs/runtime.yaml",
    figsize: tuple = (16, 4),
    show: bool = True,
):
    """
    단일 카테고리의 특정 샘플 시각화 (간편 함수)

    Args:
        model_name: 모델 이름 (patchcore, efficientad, winclip)
        dataset: 데이터셋 이름 (None이면 첫 번째 학습된 카테고리 사용)
        category: 카테고리 이름
        sample_idx: 시각화할 샘플 인덱스
        config_path: 설정 파일 경로
        figsize: figure 크기
        show: plt.show() 호출 여부

    Example:
        >>> from src.visual.plot import visualize_single_prediction
        >>> visualize_single_prediction("patchcore", "GoodsAD", "cigarette_box", sample_idx=5)
    """
    import time
    # train_anomalib.py에서 Anomalibs 클래스 임포트
    import sys
    from pathlib import Path as P
    scripts_path = P(__file__).parent.parent.parent / "scripts"
    if str(scripts_path) not in sys.path:
        sys.path.insert(0, str(scripts_path))

    from train_anomalib import Anomalibs

    # config 수정하여 모델 지정
    from src.utils.loaders import load_config
    config = load_config(config_path)
    config["anomaly"]["model"] = model_name

    runner = Anomalibs.__new__(Anomalibs)
    runner.config = config
    runner.model_name = model_name
    runner.model_params = Anomalibs.filter_none(config["anomaly"].get(model_name, {}))
    runner.training_config = Anomalibs.filter_none(config.get("training", {}))
    runner.data_root = P(config["data"]["root"])
    runner.output_root = P(config["data"]["output_root"])
    runner.output_config = config.get("output", {})
    runner.engine_config = config.get("engine", {})

    from src.utils.device import get_device
    from src.datasets.dataloader import MMADLoader
    runner.device = get_device()
    runner.loader = MMADLoader(config=config, model_name=model_name)

    # 데이터셋/카테고리 지정 안됐으면 첫 번째 학습된 것 사용
    trained = runner.get_trained_categories()
    if not trained:
        print(f"No trained categories found for {model_name}")
        return

    if dataset is None or category is None:
        dataset, category = trained[0]
        print(f"Using first trained category: {dataset}/{category}")

    # Predict
    start_time = time.time()
    predictions = runner.predict(dataset, category, save_json=False)
    inference_time = time.time() - start_time

    n_images = sum(len(batch.image_path) for batch in predictions)
    print(f"Inference Time: {inference_time:.2f}s ({n_images} images)")

    # 특정 샘플 찾기
    current_idx = 0
    target_batch = None
    target_i = None

    for batch in predictions:
        batch_size = len(batch.image_path)
        if current_idx + batch_size > sample_idx:
            target_batch = batch
            target_i = sample_idx - current_idx
            break
        current_idx += batch_size

    if target_batch is None:
        print(f"Sample index {sample_idx} out of range (total: {n_images})")
        return

    # 시각화
    visualize_anomaly_prediction(
        image_path=target_batch.image_path[target_i],
        anomaly_map=target_batch.anomaly_map[target_i] if target_batch.anomaly_map is not None else None,
        pred_mask=target_batch.pred_mask[target_i] if target_batch.pred_mask is not None else None,
        gt_mask=target_batch.gt_mask[target_i] if target_batch.gt_mask is not None else None,
        pred_score=float(target_batch.pred_score[target_i]) if target_batch.pred_score is not None else None,
        pred_label=int(target_batch.pred_label[target_i]) if target_batch.pred_label is not None else None,
        figsize=figsize,
        title=f"{dataset} / {category} (sample #{sample_idx})",
        show=show,
    )



