# PatchCore Training Pipeline

독립적인 PatchCore 학습/평가 파이프라인입니다. anomalib을 사용하지 않고 순수 PyTorch로 구현되었습니다.

## 폴더 구조

```
patchcore_training/
├── config/
│   └── config.yaml          # 설정 파일
├── src/
│   ├── __init__.py
│   ├── dataset.py           # 데이터 로딩
│   ├── model.py             # PatchCore 모델
│   ├── trainer.py           # 학습 로직
│   ├── evaluator.py         # 평가 로직
│   └── utils.py             # 유틸리티
├── scripts/
│   ├── train.py             # 학습 스크립트
│   ├── evaluate.py          # 평가 스크립트
│   └── inference.py         # LLM용 추론 스크립트
├── checkpoints/             # 학습된 모델 저장
└── README.md
```

## 설치

별도의 설치가 필요 없습니다. 기본 PyTorch 환경에서 실행됩니다.

```bash
pip install torch torchvision numpy opencv-python scikit-learn pyyaml tqdm
```

## 설정

`config/config.yaml` 파일에서 경로 및 하이퍼파라미터를 수정합니다:

```yaml
data:
  root: "/Volumes/T7/Dataset/MMAD"  # 데이터 경로
  mmad_json: "/Volumes/T7/Dataset/MMAD/mmad_10classes.json"  # 추론용 JSON

  datasets:
    MVTec-LOCO:
      - breakfast_box
      - juice_bottle
      - pushpins
      - screw_bag
    GoodsAD:
      - cigarette_box
      - drink_bottle
      # ... 모든 클래스

model:
  backbone: "wide_resnet50_2"
  layers: ["layer2", "layer3"]
  coreset_ratio: 0.01
  n_neighbors: 9

output:
  checkpoint_dir: "patchcore_training/checkpoints"
  save_pt: true
  save_onnx: true
```

## 사용법

### 1. 학습

```bash
# 모든 카테고리 학습
python patchcore_training/scripts/train.py

# 특정 카테고리만 학습
python patchcore_training/scripts/train.py --dataset GoodsAD --category cigarette_box

# 커스텀 config 사용
python patchcore_training/scripts/train.py --config path/to/config.yaml
```

### 2. 평가

```bash
# 모든 학습된 모델 평가
python patchcore_training/scripts/evaluate.py

# 특정 카테고리만 평가
python patchcore_training/scripts/evaluate.py --dataset GoodsAD --category cigarette_box

# 결과 JSON 저장
python patchcore_training/scripts/evaluate.py --output output/eval_results.json
```

### 3. LLM 평가용 추론

mmad_10classes.json의 모든 이미지에 대해 PatchCore 추론을 실행하고,
LLM 평가에 사용할 JSON 파일을 생성합니다.

```bash
# 전체 추론
python patchcore_training/scripts/inference.py

# 테스트 (100개 이미지만)
python patchcore_training/scripts/inference.py --max-images 100

# 결과 파일 지정
python patchcore_training/scripts/inference.py --output output/patchcore_predictions.json

# 이전 결과에서 이어서 실행
python patchcore_training/scripts/inference.py --resume
```

## 출력 형식

### 체크포인트

각 카테고리별로 저장됩니다:
- `checkpoints/{dataset}/{category}/model.pt` - PyTorch 체크포인트
- `checkpoints/{dataset}/{category}/model.onnx` - ONNX 모델

### 추론 결과 (LLM 평가용)

```json
{
  "image_path": "GoodsAD/cigarette_box/test/defect/001.jpg",
  "anomaly_score": 0.8234,
  "is_anomaly": true,
  "threshold": 0.5,
  "defect_location": {
    "has_defect": true,
    "region": "center-right",
    "bbox": [120, 80, 200, 160],
    "center": [160.0, 120.0],
    "area_ratio": 0.0523
  },
  "map_stats": {
    "max": 0.9123,
    "mean": 0.2341,
    "std": 0.1823
  },
  "metadata": {
    "dataset": "GoodsAD",
    "class_name": "cigarette_box",
    "model_type": "patchcore"
  }
}
```

## LLM 평가와 연동

추론 결과를 LLM 평가에 사용하려면:

```bash
# 1. PatchCore 추론
python patchcore_training/scripts/inference.py \
    --output output/patchcore_predictions.json

# 2. LLM 평가 (--with-ad 옵션 사용)
python scripts/eval_llm_baseline.py \
    --model llava \
    --with-ad \
    --ad-output output/patchcore_predictions.json \
    --output-dir output/eval/llava_with_patchcore
```

## 데이터셋 구조

### MVTec-LOCO
```
MVTec-LOCO/
└── breakfast_box/
    ├── train/
    │   └── good/
    ├── test/
    │   ├── good/
    │   └── {defect_type}/
    └── ground_truth/
        └── {defect_type}/
            └── {image_name}_mask.png  # _mask 접미사
```

### GoodsAD
```
GoodsAD/
└── cigarette_box/
    ├── train/
    │   └── good/
    ├── test/
    │   ├── good/
    │   └── {defect_type}/
    └── ground_truth/
        └── {defect_type}/
            └── {image_name}.png  # 접미사 없음
```

## 주의사항

- 이 파이프라인은 다른 폴더를 참조하지 않습니다
- anomalib과 독립적으로 동작합니다
- GPU 사용을 권장합니다 (CPU도 가능하나 느림)
