from __future__ import annotations

import platform
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import rc
import seaborn as sns
from PIL import Image


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


def anomaly_image_plot(
    image_path: str | Path,
    figsize: tuple = (16, 4),
    title: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Anomalib이 저장한 단일 결과 이미지 시각화

    Args:
        image_path: Anomalib 결과 이미지 경로
        figsize: figure 크기
        title: 제목 (None이면 파일명 사용)
        show: True면 plt.show() 호출

    Returns:
        matplotlib Figure 객체

    Example:
        >>> from src.visual.plot import anomaly_image_plot
        >>> anomaly_image_plot("results/Patchcore/MVTecAD/bottle/v0/images/broken_small/000.png")
    """
    image_path = Path(image_path)
    img = Image.open(image_path)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title if title else f"{image_path.parent.name}/{image_path.name}")

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def anomaly_grid_from_dir(
    result_dir: str | Path,
    n_cols: int = 2,
    n_samples: int | None = None,
    figsize_per_image: tuple = (8, 2),
    categories: list | None = None,
    title: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Anomalib 결과 폴더에서 이미지를 로드하여 그리드로 시각화

    Args:
        result_dir: Anomalib 결과 이미지 폴더 경로 (예: results/Patchcore/MVTecAD/bottle/v0/images)
        n_cols: 열 개수
        n_samples: 표시할 샘플 수 (None이면 전체)
        figsize_per_image: 이미지당 크기
        categories: 표시할 카테고리 리스트 (None이면 전체, 예: ['good', 'broken_small'])
        title: 전체 제목
        show: True면 plt.show() 호출

    Returns:
        matplotlib Figure 객체

    Example:
        >>> from src.visual.plot import anomaly_grid_from_dir
        >>> anomaly_grid_from_dir("results/Patchcore/MVTecAD/bottle/v0/images", n_samples=8)
        >>> anomaly_grid_from_dir("results/...", categories=['good', 'broken_small'], n_samples=4)
    """
    result_dir = Path(result_dir)

    # 이미지 파일 수집
    all_images = []
    if categories:
        for cat in categories:
            cat_dir = result_dir / cat
            if cat_dir.exists():
                all_images.extend(sorted(cat_dir.glob("*.png")))
    else:
        all_images = sorted(result_dir.glob("*/*.png"))

    if not all_images:
        print(f"No images found in {result_dir}")
        return None

    # 샘플 수 제한
    total = len(all_images)
    if n_samples is not None:
        total = min(n_samples, total)
    all_images = all_images[:total]

    n_rows = (total + n_cols - 1) // n_cols
    figsize = (figsize_per_image[0] * n_cols, figsize_per_image[1] * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, img_path in enumerate(all_images):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]

        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(f"{img_path.parent.name}/{img_path.name}", fontsize=8)
        ax.axis("off")
        # 이미지 간 경계선 추가
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('gray')
            spine.set_linewidth(2)

    # 빈 칸 숨기기
    for i in range(total, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def anomaly_compare_categories(
    result_dir: str | Path,
    categories: list | None = None,
    n_per_category: int = 2,
    figsize_per_image: tuple = (5, 1.5),
    title: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    카테고리별로 Anomalib 결과 이미지 비교 시각화

    Args:
        result_dir: Anomalib 결과 이미지 폴더 경로
        categories: 비교할 카테고리 리스트 (None이면 전체)
        n_per_category: 카테고리당 표시할 이미지 수
        figsize_per_image: 이미지당 크기
        title: 전체 제목
        show: True면 plt.show() 호출

    Returns:
        matplotlib Figure 객체

    Example:
        >>> from src.visual.plot import anomaly_compare_categories
        >>> anomaly_compare_categories("results/.../images", categories=['good', 'broken_large'])
    """
    result_dir = Path(result_dir)

    # 카테고리 폴더 찾기
    if categories is None:
        categories = [d.name for d in result_dir.iterdir() if d.is_dir()]

    n_rows = len(categories)
    n_cols = n_per_category
    figsize = (figsize_per_image[0] * n_cols, figsize_per_image[1] * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for row, cat in enumerate(categories):
        cat_dir = result_dir / cat
        if not cat_dir.exists():
            continue

        images = sorted(cat_dir.glob("*.png"))[:n_per_category]

        for col, img_path in enumerate(images):
            ax = axes[row, col]
            img = Image.open(img_path)
            ax.imshow(img)
            if col == 0:
                ax.set_ylabel(cat, fontsize=10, rotation=0, ha='right', va='center')
            ax.set_title(img_path.name, fontsize=8)
            ax.axis("off")

        # 빈 칸 숨기기
        for col in range(len(images), n_cols):
            axes[row, col].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    if show:
        plt.show()
    return fig
