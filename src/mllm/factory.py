"""LLM client factory — shared registry and instantiation logic.

Extracted from scripts/eval_llm_baseline.py so that both the eval script
and the experiment runner can reuse the same code.
"""
from __future__ import annotations

from .base import BaseLLMClient

# Model registry with HuggingFace model IDs
MODEL_REGISTRY = {
    # API models - OpenAI
    "gpt-4o": {"type": "api", "class": "GPT4Client", "model": "gpt-4o"},
    "gpt-4o-mini": {"type": "api", "class": "GPT4Client", "model": "gpt-4o-mini"},
    "gpt-4v": {"type": "api", "class": "GPT4Client", "model": "gpt-4-vision-preview"},

    # API models - Anthropic
    "claude": {"type": "api", "class": "ClaudeClient", "model": "claude-sonnet-4-20250514"},
    "claude-sonnet": {"type": "api", "class": "ClaudeClient", "model": "claude-sonnet-4-20250514"},
    "claude-haiku": {"type": "api", "class": "ClaudeClient", "model": "claude-3-5-haiku-20241022"},

    # API models - Google Gemini (FREE tier available!)
    "gemini": {"type": "api", "class": "GeminiClient", "model": "gemini-1.5-flash"},
    "gemini-flash": {"type": "api", "class": "GeminiClient", "model": "gemini-1.5-flash"},
    "gemini-pro": {"type": "api", "class": "GeminiClient", "model": "gemini-1.5-pro"},
    "gemini-2.0-flash": {"type": "api", "class": "GeminiClient", "model": "gemini-2.0-flash-exp"},

    # Qwen models
    "qwen": {"type": "local", "class": "QwenVLClient", "model": "Qwen/Qwen2.5-VL-7B-Instruct"},
    "qwen-7b": {"type": "local", "class": "QwenVLClient", "model": "Qwen/Qwen2.5-VL-7B-Instruct"},
    "qwen-2b": {"type": "local", "class": "QwenVLClient", "model": "Qwen/Qwen2.5-VL-2B-Instruct"},
    "qwen2-vl": {"type": "local", "class": "QwenVLClient", "model": "Qwen/Qwen2-VL-7B-Instruct"},
    "qwen3-vl-8b": {"type": "local", "class": "QwenVLClient", "model": "Qwen/Qwen3-VL-8B-Instruct"},

    # InternVL models
    "internvl": {"type": "local", "class": "InternVLClient", "model": "OpenGVLab/InternVL2-8B"},
    "internvl-8b": {"type": "local", "class": "InternVLClient", "model": "OpenGVLab/InternVL2-8B"},
    "internvl-4b": {"type": "local", "class": "InternVLClient", "model": "OpenGVLab/InternVL2-4B"},
    "internvl-2b": {"type": "local", "class": "InternVLClient", "model": "OpenGVLab/InternVL2-2B"},
    "internvl-1b": {"type": "local", "class": "InternVLClient", "model": "OpenGVLab/InternVL2-1B"},
    "internvl2.5-8b": {"type": "local", "class": "InternVLClient", "model": "OpenGVLab/InternVL2_5-8B"},

    # LLaVA models
    "llava": {"type": "local", "class": "LLaVAClient", "model": "llava-hf/llava-1.5-7b-hf"},
    "llava-7b": {"type": "local", "class": "LLaVAClient", "model": "llava-hf/llava-1.5-7b-hf"},
    "llava-13b": {"type": "local", "class": "LLaVAClient", "model": "llava-hf/llava-1.5-13b-hf"},
    "llava-v1.6-7b": {"type": "local", "class": "LLaVAClient", "model": "llava-hf/llava-v1.6-mistral-7b-hf"},
    "llava-onevision": {"type": "local", "class": "LLaVAClient", "model": "llava-hf/llava-onevision-qwen2-7b-ov-hf"},
}


def list_llm_models() -> list[str]:
    """Return sorted list of available model names."""
    return sorted(MODEL_REGISTRY.keys())


def get_llm_client(model_name: str, model_path: str = None, **kwargs) -> BaseLLMClient:
    """Factory function to get LLM client by name."""
    model_lower = model_name.lower()

    # Check registry first
    if model_lower in MODEL_REGISTRY:
        info = MODEL_REGISTRY[model_lower]
        actual_model = model_path or info["model"]

        if info["class"] == "GPT4Client":
            from .openai_client import GPT4Client
            return GPT4Client(model=actual_model, **kwargs)

        elif info["class"] == "ClaudeClient":
            from .claude_client import ClaudeClient
            return ClaudeClient(model=actual_model, **kwargs)

        elif info["class"] == "GeminiClient":
            from .gemini_client import GeminiClient
            return GeminiClient(model=actual_model, **kwargs)

        elif info["class"] == "QwenVLClient":
            from .qwen_client import QwenVLClient
            return QwenVLClient(model_path=actual_model, **kwargs)

        elif info["class"] == "InternVLClient":
            from .internvl_client import InternVLClient
            return InternVLClient(model_path=actual_model, **kwargs)

        elif info["class"] == "LLaVAClient":
            from .llava_client import LLaVAClient
            return LLaVAClient(model_path=actual_model, **kwargs)

    # Allow direct HuggingFace model paths
    if "/" in model_name:
        model_lower_path = model_name.lower()
        if "qwen" in model_lower_path:
            from .qwen_client import QwenVLClient
            return QwenVLClient(model_path=model_name, **kwargs)
        elif "internvl" in model_lower_path:
            from .internvl_client import InternVLClient
            return InternVLClient(model_path=model_name, **kwargs)
        elif "llava" in model_lower_path:
            from .llava_client import LLaVAClient
            return LLaVAClient(model_path=model_name, **kwargs)

    raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
