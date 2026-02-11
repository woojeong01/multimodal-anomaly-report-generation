# Experiment Runner System

AD 모델(patchcore/winclip/efficientad)과 LLM(qwen/llava/gpt-4o/claude/gemini) 조합별 MMAD 평가를 자동화하는 시스템.
YAML 설정에서 `ad_model`과 `llm` 두 줄만 바꾸면 실험이 실행되고, 결과가 자동 저장되어 비교 테이블로 확인 가능하다.

---

## 빠른 시작

```bash
# 1. YAML 설정 수정 후 실행
python scripts/run_experiment.py

# 2. CLI 오버라이드로 실행
python scripts/run_experiment.py --llm qwen --ad-model null
python scripts/run_experiment.py --llm gpt-4o --ad-model patchcore

# 3. 빠른 테스트 (이미지 5장만)
python scripts/run_experiment.py --llm qwen --max-images 5

# 4. 사용 가능한 모델 확인
python scripts/run_experiment.py --list-models

# 5. 결과 비교
python scripts/compare_results.py
python scripts/compare_results.py --sort accuracy
```

---

## 파일 구조

```
configs/
  experiment.yaml          # 실험 설정 파일 (ad_model, llm, few_shot 등)

src/
  mllm/
    factory.py             # MODEL_REGISTRY + get_llm_client() 공용 모듈
    __init__.py            # factory 모듈 export 포함
  config/
    __init__.py
    experiment.py          # ExperimentConfig dataclass + YAML 로더

scripts/
  run_experiment.py        # 메인 실험 러너
  compare_results.py       # 결과 비교 테이블
  eval_llm_baseline.py     # (수정됨) factory.py에서 import

outputs/eval/
  answers_*.json           # 실험 결과 (질문별 답변)
  answers_*.meta.json      # 실험 메타데이터 (정확도, 설정, 시간 등)
```

---

## 설정 파일

### `configs/experiment.yaml`

```yaml
ad_model: null          # patchcore | winclip | efficientad | null (AD 없이)
llm: qwen               # qwen | llava | gpt-4o | claude | gemini | internvl | ...

eval:
  few_shot: 1            # few-shot 예제 수 (0~8)
  similar_template: true  # true: 유사 템플릿 / false: 랜덤 템플릿
  max_images: null        # null = 전체 / 숫자 = 빠른 테스트용
  max_image_size: [512, 512]  # LLM에 전달할 이미지 최대 크기 [width, height]
  batch_mode: false       # true: 한 번에 질문, false: 개별 질문
  resume: false           # true: 기존 결과 이어서 진행
```

`max_image_size`는 `BaseLLMClient`에 전달되어, 이미지가 이 크기를 초과하면 비율을 유지한 채 리사이즈한다. 기본값은 `[512, 512]`.

YAML 수정 없이 CLI 인자로도 오버라이드 가능:

| CLI 인자 | YAML 필드 | 예시 |
|----------|-----------|------|
| `--llm` | `llm` | `--llm gpt-4o` |
| `--ad-model` | `ad_model` | `--ad-model patchcore` |
| `--few-shot` | `eval.few_shot` | `--few-shot 3` |
| `--max-images` | `eval.max_images` | `--max-images 10` |
| `--data-root` | `data_root` | `--data-root /data/MMAD` |
| `--output-dir` | `output_dir` | `--output-dir outputs/my_exp` |
| `--resume` | `eval.resume` | `--resume` |
| `--config` | - | `--config configs/my_exp.yaml` |

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

| 이름 | 모델 ID |
|------|---------|
| `qwen` | Qwen/Qwen2.5-VL-7B-Instruct |
| `qwen-2b` | Qwen/Qwen2.5-VL-2B-Instruct |
| `qwen3-vl-8b` | Qwen/Qwen3-VL-8B-Instruct |
| `internvl` | OpenGVLab/InternVL2-8B |
| `llava` | llava-hf/llava-1.5-7b-hf |
| `llava-onevision` | llava-hf/llava-onevision-qwen2-7b-ov-hf |

HuggingFace 모델 경로를 직접 지정할 수도 있다: `--llm Qwen/Qwen2.5-VL-2B-Instruct`

---

## 실험 결과

### 출력 파일

실험 실행 시 `outputs/eval/` 에 두 파일이 생성된다:

**`answers_1_shot_qwen_Similar_template.json`** — 이미지별 질문/답변:
```json
[
  {
    "image": "DS-MVTec/bottle/test/broken_large/000.png",
    "question": "Is there any defect?",
    "question_type": "Anomaly Detection",
    "correct_answer": "B",
    "gpt_answer": "B"
  }
]
```

**`answers_1_shot_qwen_Similar_template.meta.json`** — 실험 메타데이터:
```json
{
  "experiment_name": "no_ad_qwen_1shot",
  "llm": "qwen",
  "ad_model": null,
  "few_shot": 1,
  "accuracy": 45.2,
  "processed": 100,
  "errors": 3,
  "elapsed_seconds": 1234.5,
  "timestamp": "2026-02-09T15:30:00"
}
```

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
patchcore_gpt-4o_1shot    gpt-4o         patchcore      1        62.8%    100    1      890s     2026-02-09T16:45
============================================================

Total experiments: 2
Best accuracy: 62.8% (patchcore_gpt-4o_1shot)
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
   |-- get_llm_client()  (src/mllm/factory.py)
   |-- load MMAD data
   |-- 이미지 순회 + LLM 호출
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
| `ExperimentConfig` | dataclass — ad_model, llm, few_shot, max_images 등 |
| `ExperimentConfig.experiment_name` | `{ad_model}_{llm}_{few_shot}shot` 자동 생성 프로퍼티 |
| `load_experiment_config(path)` | YAML 로드 + `${ENV_VAR:-default}` 확장 |

---

## 실험 시나리오 예시

```bash
# LLM만 비교 (AD 없이)
python scripts/run_experiment.py --llm qwen
python scripts/run_experiment.py --llm gpt-4o
python scripts/run_experiment.py --llm claude
python scripts/run_experiment.py --llm gemini

# AD + LLM 조합 비교
python scripts/run_experiment.py --llm qwen --ad-model patchcore
python scripts/run_experiment.py --llm qwen --ad-model efficientad
python scripts/run_experiment.py --llm gpt-4o --ad-model winclip

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

# 새 방식 — YAML + CLI 오버라이드
python scripts/run_experiment.py --llm qwen --max-images 5
```

차이점:

| | eval_llm_baseline.py | run_experiment.py |
|-|---------------------|-------------------|
| 설정 방식 | CLI 인자만 | YAML + CLI 오버라이드 |
| 메타데이터 저장 | X | `.meta.json` 자동 저장 |
| 결과 비교 | 수동 | `compare_results.py`로 자동 |
| 실험 이름 | 파일명으로 구분 | `experiment_name` 자동 생성 |
