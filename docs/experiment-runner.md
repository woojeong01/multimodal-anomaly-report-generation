# Experiment Runner System

AD 모델(patchcore)과 LLM(qwen/llava/internvl/gpt-4o/claude/gemini) 조합별 MMAD 평가를 자동화하는 시스템.
YAML 설정에서 `ad_model`과 `llm` 두 줄만 바꾸면, AD 추론부터 LLM 평가까지 한 번에 실행되고 결과가 자동 저장된다.

---

## 빠른 시작

```bash
# 1. YAML 설정 수정 후 실행
python scripts/run_experiment.py

# 2. LLM만 (AD 없이)
python scripts/run_experiment.py --llm qwen --ad-model null

# 3. AD + LLM (AD 추론 자동 실행)
python scripts/run_experiment.py --llm qwen --ad-model patchcore

# 4. 기존 AD 예측 JSON을 직접 지정 (AD 추론 스킵)
python scripts/run_experiment.py --llm qwen --ad-model patchcore --ad-output output/patchcore_predictions.json

# 5. 빠른 테스트 (이미지 5장만)
python scripts/run_experiment.py --llm qwen --max-images 5

# 5-1. 폴더별 균등 샘플링 (폴더당 3장)
python scripts/run_experiment.py --llm qwen --sample-per-folder 3

# 6. 사용 가능한 모델 확인
python scripts/run_experiment.py --list-models

# 7. 결과 비교
python scripts/compare_results.py
python scripts/compare_results.py --sort accuracy
```

---

## 파일 구조

```
configs/
  experiment.yaml          # 실험 설정 파일 (ad_model, llm, few_shot, ad 설정 등)

src/
  mllm/
    factory.py             # MODEL_REGISTRY + get_llm_client() 공용 모듈
    __init__.py            # factory 모듈 export 포함
  config/
    __init__.py
    experiment.py          # ExperimentConfig dataclass + YAML 로더

scripts/
  run_experiment.py        # 메인 실험 러너 (AD 추론 + LLM 평가)
  run_ad_inference_ckpt.py # AD 모델 추론 (anomalib .ckpt 체크포인트)
  compare_results.py       # 결과 비교 테이블
  eval_llm_baseline.py     # (수정됨) factory.py에서 import

outputs/eval/
  answers_*.json           # 실험 결과 (질문별 답변)
  answers_*.meta.json      # 실험 메타데이터 (정확도, 설정, 시간 등)
  patchcore_predictions.json  # AD 모델 추론 결과 (자동 생성)
```

---

## 설정 파일

### `configs/experiment.yaml`

```yaml
ad_model: patchcore     # patchcore | null (AD 없이)
llm: qwen               # qwen | internvl | llava | gpt-4o | claude | gemini | ...

eval:
  few_shot: 1            # few-shot 예제 수 (0~8)
  similar_template: true  # true: 유사 템플릿 / false: 랜덤 템플릿
  max_images: null        # null = 전체 / 숫자 = 빠른 테스트용
  sample_per_folder: 3    # 폴더(dataset/category/split)별 N장 샘플링 (null = 전체)
  sample_seed: 42         # 샘플링 시드 (재현성)
  max_image_size: [512, 512]  # LLM에 전달할 이미지 최대 크기 [width, height]
  batch_mode: false       # true: 한 번에 질문, false: 개별 질문
  resume: false           # true: 기존 결과 이어서 진행

# AD model inference settings (ad_model이 null이 아닐 때 사용)
ad:
  config: configs/anomaly.yaml                     # run_ad_inference_ckpt.py 설정 파일
  output: null                                     # 기존 예측 JSON (있으면 inference 스킵)
  threshold: 0.5                                   # anomaly threshold
  checkpoint_dir: /path/to/checkpoints/patchcore_anomalib  # 체크포인트 루트
  version: null                                    # 체크포인트 버전 (null = 최신, 1, 2, 3...)
```

`max_image_size`는 `BaseLLMClient`에 전달되어, 이미지가 이 크기를 초과하면 비율을 유지한 채 리사이즈한다. 기본값은 `[512, 512]`.

### 모델별 기본값

일부 모델은 `batch_mode`의 기본값이 자동으로 설정된다. CLI `--batch-mode`로 명시하면 오버라이드 가능.

| 모델 | batch_mode 기본값 | 비고 |
|------|-------------------|------|
| `llava` | `false` | 배치 모드 미지원 |
| `llava-onevision` | `false` | 배치 모드 미지원 |
| 그 외 | YAML 설정값 | `experiment.yaml`의 `eval.batch_mode` 사용 |

### CLI 오버라이드

YAML 수정 없이 CLI 인자로도 오버라이드 가능:

| CLI 인자 | YAML 필드 | 예시 |
|----------|-----------|------|
| `--llm` | `llm` | `--llm gpt-4o` |
| `--ad-model` | `ad_model` | `--ad-model patchcore` |
| `--ad-output` | `ad.output` | `--ad-output output/predictions.json` |
| `--ad-version` | `ad.version` | `--ad-version 2` |
| `--few-shot` | `eval.few_shot` | `--few-shot 3` |
| `--max-images` | `eval.max_images` | `--max-images 10` |
| `--sample-per-folder` | `eval.sample_per_folder` | `--sample-per-folder 5` |
| `--sample-seed` | `eval.sample_seed` | `--sample-seed 123` |
| `--batch-mode` | `eval.batch_mode` | `--batch-mode true` |
| `--data-root` | `data_root` | `--data-root /data/MMAD` |
| `--output-dir` | `output_dir` | `--output-dir outputs/my_exp` |
| `--resume` | `eval.resume` | `--resume` |
| `--config` | - | `--config configs/my_exp.yaml` |

---

## AD 모델 연동

### 동작 방식

`ad_model`이 설정되면 `run_experiment.py`가 자동으로 AD 추론을 실행하고 결과를 LLM에 전달한다.

```
experiment.yaml: ad_model: patchcore
        |
        v
  1. MMAD json 로드 + 샘플링 (sample_per_folder, max_images)
        |
        v
  2. 샘플링된 이미지만 담은 _sampled_mmad.json 생성
        |
        v
  3. run_ad_inference(_sampled_mmad.json)
        +-- ad.output으로 기존 JSON을 명시적으로 지정했으면 → 바로 사용
        +-- 그 외 → inference.py 자동 실행 → JSON 생성 (샘플링된 이미지만 처리)
        |
        v
  4. load_ad_predictions() → {image_path: ad_info} 딕셔너리
        |
        v
  5. 이미지별 ad_info → LLM generate_answers(ad_info=ad_info)
```

### AD 추론 실행 조건

- `ad.output` 또는 `--ad-output`으로 기존 JSON을 **명시적으로 지정**한 경우에만 스킵
- 그 외에는 **매번 inference.py를 새로 실행**하여 predictions JSON을 생성한다 (입력 데이터가 바뀌어도 항상 최신 결과 보장)

### AD 예측 JSON 형식

`run_ad_inference_ckpt.py`가 생성하는 JSON (이미지당 1개). 모델의 post_processor가 0~1로 정규화한 값:
```json
{
  "image_path": "GoodsAD/cigarette_box/test/bad/001.jpg",
  "anomaly_score": 0.87,
  "is_anomaly": true,
  "defect_location": {
    "has_defect": true,
    "region": "center",
    "bbox": [100, 50, 200, 150],
    "center": [0.75, 0.5],
    "area_ratio": 0.12,
    "confidence": 0.74
  },
  "map_stats": {
    "max": 0.74,
    "mean": 0.39,
    "std": 0.11
  }
}
```

이 정보가 LLM의 프롬프트에 `ad_info`로 포함되어, LLM이 anomaly 위치/점수를 참고하여 답변한다.

### 체크포인트 구조

anomalib `.ckpt` 형식. 버전 디렉토리(`v1`, `v2`, ...)로 관리된다.

```
checkpoints/patchcore_anomalib/
  Patchcore/
    GoodsAD/
      cigarette_box/
        v1/model.ckpt
        v2/model.ckpt    # 버전 추가 시 자동 인식
      drink_bottle/
        v1/model.ckpt
      ...
    MVTec-LOCO/
      breakfast_box/
        v1/model.ckpt
      ...
```

- `ad.version: null` → 가장 높은 버전 자동 선택
- `ad.version: 1` → v1 고정 사용
- 카테고리 목록은 `configs/anomaly.yaml`에서 관리

### 첫 실행 시 참고사항

anomalib PatchCore는 backbone(ResNet 등)을 HuggingFace Hub에서 다운로드한다. **첫 실행 시 인터넷 연결이 필요**하며, 다운로드 후에는 로컬 캐시에 저장되어 이후에는 오프라인에서도 동작한다.

Colab에서 `Failed to load model: ... cannot find the requested files in the local cache` 에러가 발생하면:
1. 인터넷 연결 확인: `!curl -s -o /dev/null -w "%{http_code}" https://huggingface.co` → `200`
2. 수동으로 backbone 캐시: 셀에서 `from anomalib.models import Patchcore; Patchcore.load_from_checkpoint("체크포인트경로", map_location="cpu")` 실행
3. 캐시 완료 후 다시 `run_experiment.py` 실행

`Warning: You are sending unauthenticated requests to the HF Hub` 경고는 무시해도 된다. backbone이 캐시되면 이후 실행에서는 표시되지 않는다.

---

## 사용 가능한 모델

`python scripts/run_experiment.py --list-models` 실행 시 전체 목록 출력.

### API 모델

| 이름 | 제공자 | 모델 ID |
|------|--------|---------|
| `gpt-4o` | OpenAI | gpt-4o |
| `gpt-4o-mini` | OpenAI | gpt-4o-mini |
| `claude` | Anthropic | claude-sonnet-4-20250514 |
| `claude-haiku` | Anthropic | claude-3-5-haiku-20241022 |
| `gemini` | Google | gemini-1.5-flash |
| `gemini-pro` | Google | gemini-1.5-pro |

### 로컬 모델 (HuggingFace)

| 이름 | 모델 ID | 비고 |
|------|---------|------|
| `qwen` | Qwen/Qwen2.5-VL-7B-Instruct | |
| `qwen-2b` | Qwen/Qwen2.5-VL-2B-Instruct | |
| `qwen3-vl-8b` | Qwen/Qwen3-VL-8B-Instruct | |
| `internvl` | OpenGVLab/InternVL3_5-8B | **기본값 (v3.5)** |
| `internvl3.5-8b` | OpenGVLab/InternVL3_5-8B | internvl과 동일 |
| `internvl2.5-8b` | OpenGVLab/InternVL2_5-8B | |
| `internvl-8b` | OpenGVLab/InternVL2-8B | v2 |
| `llava` | llava-hf/llava-1.5-7b-hf | |
| `llava-onevision` | llava-hf/llava-onevision-qwen2-7b-ov-hf | |

HuggingFace 모델 경로를 직접 지정할 수도 있다: `--llm Qwen/Qwen2.5-VL-2B-Instruct`

> **InternVL3.5 참고:** `transformers>=4.52.1` 필요 (5.x는 호환 안 됨, `pip install transformers==4.52.4` 권장).

---

## 실험 결과

### 출력 파일

실험 실행 시 `outputs/eval/` 에 파일이 생성된다:

**`answers_1_shot_qwen_Similar_template_with_patchcore.json`** — 이미지별 질문/답변:
```json
[
  {
    "image": "GoodsAD/cigarette_box/test/bad/001.jpg",
    "question": "Is there any defect?",
    "question_type": "Anomaly Detection",
    "correct_answer": "B",
    "gpt_answer": "B"
  }
]
```

**`answers_1_shot_qwen_Similar_template_with_patchcore_v1_100img.meta.json`** — 실험 메타데이터:
```json
{
  "experiment_name": "patchcore_qwen_1shot",
  "llm": "qwen",
  "ad_model": "patchcore",
  "ad_version": 1,
  "few_shot": 1,
  "sample_per_folder": 3,
  "accuracy": 62.8,
  "processed": 100,
  "errors": 3,
  "elapsed_seconds": 1234.5,
  "timestamp": "2026-02-09T15:30:00"
}
```

### 출력 파일명 규칙

```
answers_{few_shot}_shot_{llm}_{template}{_with_{ad_model}}{_v{version}}_{N}img.json
```

예시:
- `answers_1_shot_qwen_Random_template_with_patchcore_v1_20img.json` — AD v1 사용
- `answers_1_shot_qwen_Random_template_with_patchcore_100img.json` — AD 버전 미지정 (latest)
- `answers_1_shot_qwen_Random_template_20img.json` — AD 없이 LLM만

### 비교 테이블

```bash
python scripts/compare_results.py
python scripts/compare_results.py --sort accuracy   # 정확도순 정렬
python scripts/compare_results.py --sort name        # 실험명순 정렬
```

출력 예시:
```
============================================================
Experiment                LLM            AD Model       Few-shot Accuracy Images Errors Time     Timestamp
------------------------------------------------------------
no_ad_qwen_1shot          qwen           none           1        45.2%    100    3      1235s    2026-02-09T15:30
patchcore_qwen_1shot      qwen           patchcore      1        58.3%    100    2      1350s    2026-02-09T16:00
patchcore_gpt-4o_1shot    gpt-4o         patchcore      1        62.8%    100    1      890s     2026-02-09T16:45
============================================================

Total experiments: 3
Best accuracy: 62.8% (patchcore_gpt-4o_1shot)
```

---

## Stratified Sampling (폴더별 샘플링)

MMAD 데이터셋은 `{dataset}/{category}/{image_type}/{split}` 구조를 갖는다. `sample_per_folder`를 설정하면 각 폴더(dataset/category/split)에서 균등하게 N장씩 샘플링한다.

### 동작 방식

```
전체 이미지 목록 (예: 5000장)
        |
        v
stratified_sample(n_per_folder=3, seed=42)
  - GoodsAD/cigarette_box/good → 3장
  - GoodsAD/cigarette_box/bad  → 3장
  - GoodsAD/drink_bottle/good  → 3장
  - GoodsAD/drink_bottle/bad   → 3장
  - ...
        |
        v
샘플링 결과 (예: 180장)
        |
        v
max_images 적용 (설정 시) → 최종 이미지 수
        |
        v
_sampled_mmad.json 생성 → AD inference + LLM 평가 모두 이 이미지만 사용
```

- 샘플링은 **AD inference 이전에** 수행된다. AD 모델도 샘플링된 이미지만 처리한다.
- `sample_per_folder`가 먼저 적용된 후, `max_images`가 그 결과에 추가로 적용된다.
- `sample_seed`로 재현성을 보장한다 (같은 시드 = 같은 샘플).
- `null`로 설정하면 샘플링 없이 전체 이미지를 사용한다.

### 사용 예시

```bash
# YAML에서 설정 (폴더당 3장)
# experiment.yaml:
#   eval:
#     sample_per_folder: 3
#     sample_seed: 42

# CLI 오버라이드
python scripts/run_experiment.py --sample-per-folder 5
python scripts/run_experiment.py --sample-per-folder 3 --sample-seed 123

# 샘플링 + max_images 조합
python scripts/run_experiment.py --sample-per-folder 5 --max-images 50
```

출력 예시:
```
Stratified sampling: 3장/폴더, 60폴더
  Total: 5000 -> Sampled: 180 (normal=90, anomaly=90)
```

---

## 아키텍처

### 파이프라인 흐름

```
configs/experiment.yaml
        |
        v
ExperimentConfig (src/config/experiment.py)
        |
        v
run_experiment.py
   |
   |-- load MMAD data
   |-- stratified_sample() + max_images → 이미지 샘플링
   |-- _sampled_mmad.json 생성 (샘플링된 이미지만)
   |
   |-- [ad_model 설정 시] run_ad_inference(_sampled_mmad.json)
   |       |-- ad.output 명시 시 → 기존 JSON 재사용
   |       |-- 그 외 → inference.py 자동 실행 (샘플링된 이미지만 처리)
   |       +-- predictions JSON → load_ad_predictions()
   |
   |-- get_llm_client()  (src/mllm/factory.py)
   |-- 이미지 순회 + LLM 호출 (ad_info 포함)
   |-- 결과 저장 (.json + .meta.json)
   |-- calculate_accuracy_mmad()  (src/eval/metrics.py)
        |
        v
compare_results.py  →  비교 테이블 출력
```

### `src/mllm/factory.py`

`eval_llm_baseline.py`에 있던 `MODEL_REGISTRY`와 `get_llm_client()`를 공용 모듈로 추출한 것.
`run_experiment.py`와 `eval_llm_baseline.py` 모두 이 모듈에서 import한다.

| 함수 | 설명 |
|------|------|
| `MODEL_REGISTRY` | 모델 이름 → {type, class, model} 매핑 딕셔너리 |
| `get_llm_client(name)` | 이름으로 LLM 클라이언트 인스턴스 생성 |
| `list_llm_models()` | 사용 가능한 모델 이름 목록 반환 |

### `src/config/experiment.py`

| 클래스/함수 | 설명 |
|-------------|------|
| `ExperimentConfig` | dataclass — ad_model, llm, few_shot, ad_config, ad_output 등 |
| `ExperimentConfig.experiment_name` | `{ad_model}_{llm}_{few_shot}shot` 자동 생성 프로퍼티 |
| `load_experiment_config(path)` | YAML 로드 + `${ENV_VAR:-default}` 확장 |

---

## 실험 시나리오 예시

```bash
# LLM만 비교 (AD 없이)
python scripts/run_experiment.py --llm qwen
python scripts/run_experiment.py --llm internvl
python scripts/run_experiment.py --llm gpt-4o
python scripts/run_experiment.py --llm claude
python scripts/run_experiment.py --llm gemini

# AD + LLM 조합 비교 (AD 추론 자동 실행)
python scripts/run_experiment.py --llm qwen --ad-model patchcore
python scripts/run_experiment.py --llm internvl --ad-model patchcore
python scripts/run_experiment.py --llm gpt-4o --ad-model patchcore

# 이미 생성된 AD 예측 JSON 사용 (추론 스킵)
python scripts/run_experiment.py --llm qwen --ad-model patchcore --ad-output output/patchcore_predictions.json

# Few-shot 수 비교
python scripts/run_experiment.py --llm qwen --few-shot 0
python scripts/run_experiment.py --llm qwen --few-shot 1
python scripts/run_experiment.py --llm qwen --few-shot 3

# 전체 결과 확인
python scripts/compare_results.py --sort accuracy
```

---

## 기존 스크립트와의 관계

`eval_llm_baseline.py`는 그대로 동작한다. 내부적으로 `MODEL_REGISTRY`와 `get_llm_client()`를 `factory.py`에서 가져오도록 변경되었을 뿐, CLI 인터페이스와 동작은 동일하다.

```bash
# 기존 방식 — 여전히 동작
python scripts/eval_llm_baseline.py --model qwen --few-shot 1 --similar-template --max-images 5

# 기존 방식으로 AD 사용 — 수동으로 2단계 실행
python patchcore_training/scripts/inference.py
python scripts/eval_llm_baseline.py --model qwen --with-ad --ad-output output/patchcore_predictions.json

# 새 방식 — YAML + CLI 오버라이드, AD 자동 실행
python scripts/run_experiment.py --llm qwen --ad-model patchcore
```

차이점:

| | eval_llm_baseline.py | run_experiment.py |
|-|---------------------|-------------------|
| 설정 방식 | CLI 인자만 | YAML + CLI 오버라이드 |
| AD 모델 연동 | 수동 (inference 별도 실행 + `--ad-output` 지정) | 자동 (inference 자동 실행 + JSON 연결) |
| 메타데이터 저장 | X | `.meta.json` 자동 저장 |
| 결과 비교 | 수동 | `compare_results.py`로 자동 |
| 실험 이름 | 파일명으로 구분 | `experiment_name` 자동 생성 |
