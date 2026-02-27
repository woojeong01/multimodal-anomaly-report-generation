# Anomalib Custom Dataset 트러블슈팅

## 문제

Anomalib은 MVTec-AD, VisA 같은 표준 데이터셋만 공식 지원한다.
MVTec-LOCO와 GoodsAD는 디렉토리 구조·파일 포맷이 달라 내장 로더로 불러오면 이미지를 0개 인식하거나 마스크를 찾지 못하는 문제가 발생했다.

| 데이터셋 | 내장 로더로 발생한 문제 |
|---|---|
| **MVTec-LOCO** | 마스크가 단일 파일이 아닌 **중첩 폴더** 구조 → `gt_mask` 로드 실패 |
| **GoodsAD** | 이미지 `.jpg` + 마스크 `.png` 혼합 확장자 → Folder Dataset이 마스크 매칭 실패 |

---

## 원인 분석

### MVTec-LOCO 마스크 구조

Anomalib 내장 로더는 마스크 경로를 `ground_truth/{defect_type}/{image_stem}.png` 한 파일로 가정한다.
MVTec-LOCO는 실제로 마스크가 폴더 안에 여러 장으로 존재한다.

```
# Anomalib 내장 로더가 기대하는 구조
ground_truth/bent_wire/000.png   ← 파일

# MVTec-LOCO 실제 구조
ground_truth/bent_wire/000/
    000.png                      ← 폴더 안에 파일
    001.png                      ← 복수 마스크
```

### GoodsAD 확장자 불일치

```
# GoodsAD 실제 구조
test/bad/000_001.jpg             ← 이미지 JPG
ground_truth/bad/000_001.png     ← 마스크 PNG (확장자 다름)
```

Anomalib `Folder` Dataset은 이미지 stem 기반으로 마스크를 찾을 때 같은 확장자를 기대하거나,
확장자 케이스 대소문자 구분 문제로 `Found 0 images` 오류가 발생했다.

---

## 해결 방법

`anomalib.data.datasets.base.image.AnomalibDataset`을 **상속**해 커스텀 Dataset을 직접 구현.

핵심은 `make_dataset()` 메서드에서 pandas DataFrame을 직접 구성해 데이터셋별 디렉토리 구조 차이를 흡수하는 것이다.

### 구현 파일

`src/datasets/dataloader.py`

---

## 구현 코드

### 1. MVTecLOCODataset — 중첩 마스크 처리

```python
from anomalib.data.datasets.base.image import AnomalibDataset
from anomalib.data.dataclasses import ImageItem

class MVTecLOCODataset(AnomalibDataset):
    def __init__(self, root, category, split="train", preprocess=None, image_size=(256, 256)):
        super().__init__(augmentations=None)
        self.root = Path(root)
        self._category = category
        self.split = split
        self._samples = self.make_dataset()
        self.preprocess = preprocess
        self.image_size = image_size

    def make_dataset(self) -> pd.DataFrame:
        samples_list = []
        test_dir = cat_path / "test"
        gt_dir = cat_path / "ground_truth"

        for defect_dir in sorted(test_dir.iterdir()):
            defect_type = defect_dir.name
            for img_path in sorted(defect_dir.glob("*.png")):
                stem = img_path.stem
                mask_path = None

                # 핵심: 폴더 안에서 첫 번째 마스크 파일 선택
                mask_dir = gt_dir / defect_type / stem
                if mask_dir.exists() and mask_dir.is_dir():
                    mask_files = sorted(mask_dir.glob("*.png"))
                    if mask_files:
                        mask_path = str(mask_files[0])

                samples_list.append({
                    "image_path": str(img_path),
                    "label": "normal" if defect_type == "good" else "anomaly",
                    "label_index": 0 if defect_type == "good" else 1,
                    "mask_path": mask_path,
                    "split": self.split,
                })

        samples = pd.DataFrame(samples_list)
        samples.attrs["task"] = "segmentation"
        return samples

    def __getitem__(self, index: int) -> ImageItem:
        sample = self.samples.iloc[index]
        image_np = read_image(sample.image_path)

        # float64 → PIL이 처리 못하므로 uint8 변환
        if image_np.dtype in [np.float32, np.float64]:
            image_np = (image_np * 255).astype(np.uint8)
        image = Image.fromarray(image_np)

        gt_mask = None
        if pd.notna(sample.mask_path) and sample.mask_path:
            gt_mask = read_mask(sample.mask_path, as_tensor=True).float()
            if gt_mask.ndim == 2:
                gt_mask = gt_mask.unsqueeze(0)
            gt_mask = torch.nn.functional.interpolate(
                gt_mask.unsqueeze(0), size=self.image_size, mode="nearest"
            ).squeeze(0)

        if self.preprocess:
            image = self.preprocess(image)

        # 마스크 없는 정상 이미지는 zeros 텐서로 패딩
        if gt_mask is None and isinstance(image, torch.Tensor):
            gt_mask = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32)

        return ImageItem(
            image_path=sample.image_path,
            image=image,
            gt_label=torch.tensor(sample.label_index, dtype=torch.long),
            gt_mask=gt_mask,
        )
```

### 2. GoodsADDataset — 혼합 확장자 처리

```python
class GoodsADDataset(AnomalibDataset):
    def make_dataset(self) -> pd.DataFrame:
        for img_path in iter_image_files(defect_dir):  # jpg/png 무관하게 순회
            mask_path = None
            if defect_type != "good":
                # 이미지 확장자와 무관하게 마스크는 항상 .png
                mask_file = gt_dir / defect_type / f"{img_path.stem}.png"
                if mask_file.exists():
                    mask_path = str(mask_file)
            ...
```

### 3. 커스텀 collate 함수

Anomalib 내장 collate가 None 마스크를 처리하지 못해 직접 구현:

```python
def collate_items(items: list) -> ImageBatch:
    images = torch.stack([item.image for item in items])
    gt_labels = torch.stack([item.gt_label for item in items])

    gt_masks = None
    if all(item.gt_mask is not None for item in items):
        gt_masks = torch.stack([item.gt_mask.float() for item in items])

    return ImageBatch(
        image_path=[item.image_path for item in items],
        image=images,
        gt_label=gt_labels,
        gt_mask=gt_masks,
    )
```

### 4. DataModule — LightningDataModule 상속

```python
from lightning.pytorch import LightningDataModule

class GoodsADDataModule(LightningDataModule):
    def setup(self, stage=None):
        self.train_data = GoodsADDataset(self.root, self.category, split="train", ...)
        self.test_data  = GoodsADDataset(self.root, self.category, split="test",  ...)

    def train_dataloader(self):
        return DataLoader(self.train_data, ..., collate_fn=collate_items)

    def test_dataloader(self):
        return DataLoader(self.test_data, ..., collate_fn=collate_items)
```

### 5. 팩토리 패턴으로 데이터셋 통합 — MMADLoader

```python
class MMADLoader:
    def get_datamodule(self, dataset: str, category: str, **kwargs):
        loaders = {
            "MVTec-AD":   self.mvtec_ad,    # anomalib 내장
            "VisA":       self.visa,         # anomalib 내장
            "MVTec-LOCO": self.mvtec_loco,  # 커스텀
            "GoodsAD":    self.goods_ad,    # 커스텀
        }
        return loaders[dataset](category, **kwargs)
```

---

## 핵심 기술 포인트 요약

| 포인트 | 설명 |
|---|---|
| **`AnomalibDataset` 상속** | anomalib 내부 파이프라인(PatchCore 학습/추론)과 인터페이스 호환 유지 |
| **`make_dataset()` 직접 구현** | pandas DataFrame으로 경로 구성 → 데이터셋별 구조 차이 흡수 |
| **`__getitem__()` 오버라이드** | float64 → uint8 변환, 마스크 없을 때 zeros 패딩 처리 |
| **커스텀 `collate_fn`** | `gt_mask=None` 케이스 안전하게 처리해 `ImageBatch` 반환 |
| **팩토리 패턴 (`MMADLoader`)** | 데이터셋 이름으로 적절한 DataModule을 반환, 호출부 코드 변경 불필요 |

---

## 관련 파일

| 파일 | 역할 |
|---|---|
| `src/datasets/dataloader.py` | `MVTecLOCODataset`, `GoodsADDataset`, `MMADLoader` 구현 |
| `src/datasets/anomalib_folder_builder.py` | MMAD CSV 레코드 → anomalib Folder 포맷 디렉토리 빌드 (심볼릭 링크) |
| `src/datasets/mmad_index_csv.py` | `MMAD_index.csv` 파싱 및 train/test 분할 |
