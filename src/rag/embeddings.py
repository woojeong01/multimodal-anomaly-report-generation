from __future__ import annotations
from src.utils.loaders import load_env


def get_embedding_model(provider: str = "huggingface", **kwargs):
    """Return a LangChain-compatible embedding model.

    Args:
        provider: One of "openai", "huggingface", "gemini".
        **kwargs: Extra keyword arguments forwarded to the embedding class.

    Returns:
        A LangChain Embeddings instance.
    """
    load_env()
    provider = provider.lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(**kwargs)

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        model_name = kwargs.pop(
            "model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        return HuggingFaceEmbeddings(model_name=model_name, **kwargs)

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        model = kwargs.pop("model", "models/embedding-001")
        return GoogleGenerativeAIEmbeddings(model=model, **kwargs)

    raise ValueError(
        f"Unknown embedding provider: {provider!r}. "
        "Choose from: openai, huggingface, gemini"
    )
