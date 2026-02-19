from __future__ import annotations

from typing import Dict, List

from .retriever import Retrievers

# ── 외부 소스(PDF + Web) 검색 품질 테스트 쿼리 ─────────────────────────────────
# PDF(PackagingGuide) 또는 CFIA_RetortPouch에서 검색되어야 하는 쿼리들
TEST_QUERIES_EXTERNAL: List[Dict] = [
    {
        "query": "pouch seal void fold hermetic integrity food package",
        "relevant_datasets": ["PackagingGuide", "CFIA_RetortPouch"],
        "description": "패키지 씰 보이드/폴드 결함",
    },
    {
        "query": "bottle cap open loose not sealed drink",
        "relevant_datasets": ["PackagingGuide"],
        "description": "음료 병 캡 미체결 결함",
    },
    {
        "query": "label mislabeled skewed misaligned food packaging",
        "relevant_datasets": ["PackagingGuide"],
        "description": "라벨 기울어짐/불일치 결함",
    },
    {
        "query": "contamination foreign material food package surface",
        "relevant_datasets": ["PackagingGuide", "CFIA_RetortPouch"],
        "description": "이물질/오염 결함",
    },
    {
        "query": "blister wrinkle delamination seal pouch serious defect",
        "relevant_datasets": ["CFIA_RetortPouch"],
        "description": "CFIA 씰 블리스터/주름 결함",
    },
    {
        "query": "missing item wrong item assembly kitting box",
        "relevant_datasets": ["PackagingGuide"],
        "description": "어셈블리 키팅 누락/오삽입 결함",
    },
    {
        "query": "swollen package bulging gas formation serious",
        "relevant_datasets": ["CFIA_RetortPouch"],
        "description": "CFIA 팽창 파우치 결함",
    },
    {
        "query": "scratch crack cosmetic surface defect packaging",
        "relevant_datasets": ["PackagingGuide"],
        "description": "외관 스크래치/크랙 결함",
    },
]

# ── 기존 MMAD JSON 지식 회귀 테스트 쿼리 ────────────────────────────────────────
# GoodsAD / MVTec-LOCO에서 검색되어야 하는 쿼리들
TEST_QUERIES_MMAD: List[Dict] = [
    {
        "query": "food package surface anomaly broken damaged",
        "relevant_datasets": ["GoodsAD"],
        "description": "GoodsAD food_package 결함",
    },
    {
        "query": "drink bottle cap open half open seal broken",
        "relevant_datasets": ["GoodsAD"],
        "description": "GoodsAD drink_bottle 캡 결함",
    },
    {
        "query": "food box surface damage deformation crushed",
        "relevant_datasets": ["GoodsAD"],
        "description": "GoodsAD food_box 변형/손상",
    },
    {
        "query": "juice bottle label anomaly structural defect",
        "relevant_datasets": ["MVTec-LOCO"],
        "description": "MVTec-LOCO juice_bottle 라벨 이상",
    },
    {
        "query": "breakfast box wrong items missing logical anomaly",
        "relevant_datasets": ["MVTec-LOCO"],
        "description": "MVTec-LOCO breakfast_box 논리적 이상",
    },
    {
        "query": "pushpins missing extra structural anomaly arrangement",
        "relevant_datasets": ["MVTec-LOCO"],
        "description": "MVTec-LOCO pushpins 구조적 이상",
    },
]


class RAGEvaluator:
    """RAG 검색 품질 평가 — Hit Rate, MRR.

    Args:
        retriever: Retrievers 인스턴스.

    Example::

        ev = RAGEvaluator(Retrievers(vectorstore))
        result = ev.evaluate(TEST_QUERIES_EXTERNAL, k=5)
        print(result["hit_rate"], result["mrr"])

        comparison = ev.compare(
            baseline=RAGEvaluator(Retrievers(baseline_vs)),
            test_queries=TEST_QUERIES_EXTERNAL,
        )
    """

    def __init__(self, retriever: Retrievers):
        self.retriever = retriever

    def evaluate(self, test_queries: List[Dict], k: int = 5) -> Dict:
        """Hit Rate와 MRR을 계산해 반환.

        Args:
            test_queries: ``[{"query", "relevant_datasets", "description"}, ...]``
            k: 쿼리당 검색할 최대 문서 수.

        Returns:
            hit_rate, mrr, n_queries, per_query 결과 포함 dict.
        """
        per_query = []
        for item in test_queries:
            docs = self.retriever.retrieve(item["query"], k=k)
            retrieved_datasets = [d.metadata.get("dataset", "") for d in docs]

            hit = any(ds in item["relevant_datasets"] for ds in retrieved_datasets)

            rr = 0.0
            for rank, ds in enumerate(retrieved_datasets, 1):
                if ds in item["relevant_datasets"]:
                    rr = 1.0 / rank
                    break

            per_query.append({
                "query": item["query"],
                "description": item["description"],
                "hit": hit,
                "reciprocal_rank": rr,
                "retrieved_datasets": retrieved_datasets,
                "relevant_datasets": item["relevant_datasets"],
            })

        n = len(per_query)
        hit_rate = sum(r["hit"] for r in per_query) / n
        mrr = sum(r["reciprocal_rank"] for r in per_query) / n

        return {
            "hit_rate": round(hit_rate, 4),
            "mrr": round(mrr, 4),
            "n_queries": n,
            "per_query": per_query,
        }

    def compare(
        self,
        baseline: RAGEvaluator,
        test_queries: List[Dict],
        k: int = 5,
        label_self: str = "Extended",
        label_baseline: str = "Baseline",
    ) -> Dict:
        """두 retriever 성능 비교.

        Args:
            baseline: 비교 기준이 되는 RAGEvaluator.
            test_queries: 공통 테스트 쿼리셋.
            k: 검색 문서 수.
            label_self: 이 evaluator의 이름 (결과 dict 키).
            label_baseline: baseline의 이름 (결과 dict 키).

        Returns:
            두 결과와 delta(차이)를 포함한 dict.
        """
        result_self = self.evaluate(test_queries, k=k)
        result_base = baseline.evaluate(test_queries, k=k)
        return {
            label_self: result_self,
            label_baseline: result_base,
            "delta": {
                "hit_rate": round(result_self["hit_rate"] - result_base["hit_rate"], 4),
                "mrr": round(result_self["mrr"] - result_base["mrr"], 4),
            },
        }
