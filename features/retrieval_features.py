from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional


class RetrievalFeatureExtractor:
    """Extracts score statistics, rank gaps, and sparse-dense agreement indicators
    from initial probe retrieval results as described in the SAGE paper.
    """

    @classmethod
    def extract(
        cls,
        fused_candidates: List[Dict[str, Any]],
        bm25_candidates: Optional[List[Dict[str, Any]]] = None,
        dense_candidates: Optional[List[Dict[str, Any]]] = None,
        agreement_k: int = 10
    ) -> Dict[str, float]:
        if not fused_candidates:
            return {
                "top1_score": 0.0,
                "score_drop": 0.0,
                "score_mean": 0.0,
                "score_std": 0.0,
                "sparse_dense_agreement": 0.0
            }

        scores = [float(item.get("score", 0.0)) for item in fused_candidates]
        top1_score = scores[0]
        top2_score = scores[1] if len(scores) > 1 else scores[0]
        score_drop = max(0.0, top1_score - top2_score)

        score_mean = float(np.mean(scores))
        score_std = float(np.std(scores)) if len(scores) > 1 else 0.0

        # Sparse-dense agreement indicator (Jaccard overlap between top-k BM25 and Dense)
        sparse_dense_agreement = 0.0
        if bm25_candidates is not None and dense_candidates is not None:
            bm25_top_ids = {c["id"] for c in bm25_candidates[:agreement_k]}
            dense_top_ids = {c["id"] for c in dense_candidates[:agreement_k]}
            union = bm25_top_ids | dense_top_ids
            intersection = bm25_top_ids & dense_top_ids
            if union:
                sparse_dense_agreement = float(len(intersection) / len(union))

        return {
            "top1_score": float(round(top1_score, 6)),
            "score_drop": float(round(score_drop, 6)),
            "score_mean": float(round(score_mean, 6)),
            "score_std": float(round(score_std, 6)),
            "sparse_dense_agreement": float(round(sparse_dense_agreement, 4))
        }

