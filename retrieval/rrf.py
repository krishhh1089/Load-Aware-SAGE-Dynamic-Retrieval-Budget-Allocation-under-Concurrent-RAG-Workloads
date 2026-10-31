from __future__ import annotations
from typing import List, Dict, Any, Optional


class ReciprocalRankFusion:
    """Reciprocal Rank Fusion (RRF) for combining sparse (BM25) and dense retrieval rankings.
    As formulated in the SAGE paper with default constant k=60.
    """

    def __init__(self, k: int = 60, bm25_weight: float = 1.0, dense_weight: float = 1.0):
        self.k = k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def fuse(
        self,
        bm25_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Fuses ranked candidate lists from BM25 and Dense retrievers."""
        scores: Dict[str, float] = {}
        passages: Dict[str, Dict[str, Any]] = {}

        # Process BM25 rankings
        for rank, item in enumerate(bm25_results, start=1):
            doc_id = item["id"]
            if doc_id not in passages:
                passages[doc_id] = dict(item)
            rrf_score = self.bm25_weight / (self.k + rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score

        # Process Dense rankings
        for rank, item in enumerate(dense_results, start=1):
            doc_id = item["id"]
            if doc_id not in passages:
                passages[doc_id] = dict(item)
            rrf_score = self.dense_weight / (self.k + rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score

        # Sort combined results by RRF score descending
        sorted_ids = sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)
        if top_k is not None:
            sorted_ids = sorted_ids[:top_k]

        fused_results = []
        for rank, doc_id in enumerate(sorted_ids, start=1):
            p = passages[doc_id]
            fused_results.append({
                "id": doc_id,
                "text": p.get("text", ""),
                "title": p.get("title", ""),
                "score": float(scores[doc_id]),
                "rank": rank,
                "retriever": "hybrid_rrf",
                "metadata": p.get("metadata", {})
            })

        return fused_results

