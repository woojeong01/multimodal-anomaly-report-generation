from __future__ import annotations
from typing import Dict, List, Optional, Union
from langchain_core.documents import Document
from langchain_chroma import Chroma


class Retrievers:
    """Semantic search over domain knowledge with optional metadata filtering."""

    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

    def retrieve(
        self,
        query: str,
        dataset: Optional[Union[str, List[str]]] = None,
        category: Optional[Union[str, List[str]]] = None,
        defect_type: Optional[str] = None,
        k: int = 3,
    ) -> List[Document]:
        """Search for relevant domain knowledge documents.

        Args:
            query: Free-text query describing the defect or context.
            dataset: Filter by dataset name(s). str or list of str.
            category: Filter by category name(s). str or list of str.
            defect_type: Exact defect type filter. anomaly 이미지에서 경로의
                defect_type을 넘기면 해당 결함 문서만 검색해 노이즈를 제거한다.
                good 이미지나 None이면 필터 없이 의미 검색만 수행한다.
            k: Number of results to return.

        Returns:
            List of matching Document objects.
        """
        where_filter = self.build_filter(dataset=dataset, category=category, defect_type=defect_type)
        kwargs: Dict = {"k": k}
        if where_filter:
            kwargs["filter"] = where_filter

        return self.vectorstore.similarity_search(query, **kwargs)

    def build_filter(
        self,
        dataset: Optional[Union[str, List[str]]] = None,
        category: Optional[Union[str, List[str]]] = None,
        defect_type: Optional[str] = None,
    ) -> Optional[Dict]:
        """Build a Chroma metadata filter dict."""
        conditions = []
        if dataset:
            if isinstance(dataset, list):
                conditions.append({"dataset": {"$in": dataset}})
            else:
                conditions.append({"dataset": dataset})
        if category:
            if isinstance(category, list):
                conditions.append({"category": {"$in": category}})
            else:
                conditions.append({"category": category})
        if defect_type:
            conditions.append({"defect_type": defect_type})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def build_query(self, category: str, defect_type: str) -> str:
        """Build a retrieval query from image path components.

        good 이미지는 결함 쿼리 대신 정상 외관 쿼리를 사용한다.
        'carpet good defect anomaly' 같은 모순된 쿼리를 방지해
        정상 이미지에서 불필요한 결함 문서가 검색되는 것을 막는다.
        """
        if defect_type == "good":
            return f"{category} normal product appearance"
        return f"{category} {defect_type} defect anomaly"

    def format_context(self, docs: List[Document]) -> str:
        """Format retrieved documents into a text block for MLLM prompts.

        Args:
            docs: List of retrieved Document objects.

        Returns:
            Formatted string ready to inject into a prompt.
        """
        if not docs:
            return "No relevant domain knowledge found."

        parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            header = (
                f"[{i}] Dataset: {meta.get('dataset', 'N/A')} | "
                f"Category: {meta.get('category', 'N/A')} | "
                f"Defect Type: {meta.get('defect_type', 'N/A')}"
            )
            parts.append(f"{header}\n{doc.page_content}")

        return "\n\n".join(parts)
