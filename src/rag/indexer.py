from __future__ import annotations
import logging
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.utils.loaders import load_json
from .embeddings import get_embedding_model

logger = logging.getLogger(__name__)
COLLECTION_NAME = "domain_knowledge"


class Indexer:
    """Load domain_knowledge.json and build / load a Chroma vector store."""

    def __init__(
        self,
        json_path: str | Path,
        persist_dir: str | Path = "vectorstore/domain_knowledge",
        embedding_provider: str = "huggingface",
        **embedding_kwargs,
    ):
        self.json_path = Path(json_path)
        self.persist_dir = Path(persist_dir)
        self.embedding = get_embedding_model(embedding_provider, **embedding_kwargs)

    def load_documents(self) -> List[Document]:
        """Parse domain_knowledge.json into LangChain Documents.

        Structure: {dataset: {category: {defect_type: description}}}

        Returns:
            List of Document objects with metadata.
        """
        data = load_json(str(self.json_path))
        docs: List[Document] = []

        for dataset, categories in data.items():
            for category, defect_types in categories.items():
                for defect_type, description in defect_types.items():
                    description = description.strip()
                    if not description:
                        continue
                    docs.append(
                        Document(
                            page_content=description,
                            metadata={
                                "dataset": dataset,
                                "category": category,
                                "defect_type": defect_type,
                            },
                        )
                    )

        logger.info("Loaded %d documents from %s", len(docs), self.json_path)
        return docs

    def build_index(self) -> Chroma:
        """Build a new Chroma vector store from the JSON and persist it."""
        docs = self.load_documents()
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding,
            collection_name=COLLECTION_NAME,
            persist_directory=str(self.persist_dir),
        )
        logger.info(
            "Built Chroma index with %d docs at %s", len(docs), self.persist_dir
        )
        return vectorstore

    def load_index(self) -> Chroma:
        """Load an existing persisted Chroma vector store."""
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embedding,
            persist_directory=str(self.persist_dir),
        )
        logger.info("Loaded Chroma index from %s", self.persist_dir)
        return vectorstore

    def get_or_create(self) -> Chroma:
        """Load existing index if available, otherwise build a new one."""
        if self.persist_dir.exists() and any(self.persist_dir.iterdir()):
            return self.load_index()
        return self.build_index()
