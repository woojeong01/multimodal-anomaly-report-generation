from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# =========================================================
# Path setup: make `src` importable even when running as a script
# =========================================================
SCRIPT_PATH = Path(__file__).resolve()
PROJ_ROOT = SCRIPT_PATH.parents[1]  # <repo_root>/
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.service.settings import load_yaml, load_runtime_config
from src.anomaly.dummy_edge import DummyEdgeAnomaly
from src.mllm.echo import EchoMLLM
from src.eval.mmad_eval import evaluate_mmad


def _resolve_path_maybe_relative(p: str | None) -> str | None:
    """
    Keep backward compatibility:
    - If user passes an absolute path, use it.
    - If relative and exists in CWD, use it.
    - Else, try relative to repo root (PROJ_ROOT).
    """
    if p is None:
        return None
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    # 1) relative to current working directory
    if pp.exists():
        return str(pp)
    # 2) fallback: relative to project root
    cand = PROJ_ROOT / pp
    if cand.exists():
        return str(cand)
    # keep original string (let downstream raise a clear error)
    return str(pp)


def _set_env_if_provided(key: str, value: str | None) -> None:
    if value is not None and str(value).strip() != "":
        os.environ[key] = str(value)


def main() -> None:
    ap = argparse.ArgumentParser()

    # --- existing behavior ---
    ap.add_argument("--config", type=str, default="configs/eval.yaml")

    # --- new: override runtime config file path ---
    ap.add_argument(
        "--runtime-config",
        type=str,
        default=None,
        help="Override runtime_config path (otherwise uses config yaml's runtime_config).",
    )

    # --- new: override runtime paths (via env vars used by anomaly.yaml) ---
    ap.add_argument("--data-root", type=str, default=None, help="Override MMAD data root (MMAD_DATA_ROOT).")
    ap.add_argument("--mmad-json", type=str, default=None, help="Override mmad.json path (MMAD_JSON_PATH).")
    ap.add_argument("--artifact-root", type=str, default=None, help="Override artifact output root (ARTIFACT_ROOT).")
    ap.add_argument("--db-path", type=str, default=None, help="Override sqlite db path (DB_PATH).")

    # --- new: override eval section values (optional) ---
    ap.add_argument("--output-dir", type=str, default=None, help="Override eval.output_dir from config.")
    ap.add_argument("--few-shot-k", type=int, default=None, help="Override eval.few_shot_k from config.")
    ap.add_argument(
        "--use-similar-template",
        type=str,
        default=None,
        choices=["true", "false", "1", "0", "yes", "no", "y", "n", "t", "f"],
        help="Override eval.use_similar_template from config (bool).",
    )
    ap.add_argument("--max-images", type=int, default=None, help="Override eval.max_images from config (int).")

    args = ap.parse_args()

    # Resolve config path (backward compatible)
    config_path = _resolve_path_maybe_relative(args.config)
    cfg = load_yaml(config_path)

    # Resolve runtime_config path:
    # - default: cfg["runtime_config"]
    # - override: --runtime-config
    runtime_config_path = args.runtime_config if args.runtime_config is not None else cfg["runtime_config"]
    runtime_config_path = _resolve_path_maybe_relative(runtime_config_path)

    # Apply path overrides as env vars (so anomaly.yaml stays unchanged)
    _set_env_if_provided("MMAD_DATA_ROOT", args.data_root)
    _set_env_if_provided("MMAD_JSON_PATH", args.mmad_json)
    _set_env_if_provided("ARTIFACT_ROOT", args.artifact_root)
    _set_env_if_provided("DB_PATH", args.db_path)

    runtime_cfg = load_runtime_config(runtime_config_path)

    # Step 1: dummy + echo (original behavior)
    anomaly = DummyEdgeAnomaly(**runtime_cfg.anomaly.get("dummy", {}))
    mllm = EchoMLLM(**runtime_cfg.mllm.get("echo", {}))

    # Eval config (original behavior, with optional CLI overrides)
    e = cfg["eval"]

    # output_dir override
    output_dir = args.output_dir if args.output_dir is not None else e["output_dir"]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "mmad_eval.json"

    # few_shot_k override
    few_shot_k = int(args.few_shot_k) if args.few_shot_k is not None else int(e.get("few_shot_k", 1))

    # use_similar_template override
    if args.use_similar_template is None:
        use_similar = bool(e.get("use_similar_template", True))
    else:
        v = args.use_similar_template.strip().lower()
        use_similar = v in ("true", "1", "yes", "y", "t")

    # max_images override
    max_images = args.max_images if args.max_images is not None else e.get("max_images", None)

    summary = evaluate_mmad(
        runtime_cfg=runtime_cfg,
        anomaly_model=anomaly,
        mllm_client=mllm,
        few_shot_k=few_shot_k,
        use_similar=use_similar,
        max_images=max_images,
        out_json=out_json,
    )

    print("=== MMAD Evaluation Summary ===")
    print(summary)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()