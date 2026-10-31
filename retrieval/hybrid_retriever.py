from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from retrieval.bm25 import BM25Retriever, ArtifactNotFoundError
from retrieval.dense_retriever import DenseRetriever
from retrieval.rrf import ReciprocalRankFusion


class HybridRetriever:
    """Combines BM25 and Dense retrieval via Reciprocal Rank Fusion (RRF)
    as defined in the original SAGE paper.
    """

    def __init__(
        self,
        bm25: Optional[BM25Retriever] = None,
        dense: Optional[DenseRetriever] = None,
        rrf: Optional[ReciprocalRankFusion] = None,
        top_candidates: int = 50,
        probe_depth: int = 2
    ):
        self.bm25 = bm25 or BM25Retriever()
        self.dense = dense or DenseRetriever()
        self.rrf = rrf or ReciprocalRankFusion(k=60)
        self.top_candidates = top_candidates
        self.probe_depth = probe_depth
        self.is_loaded = False

    def load_artifacts(
        self,
        bm25_dir: str = "artifacts/indexes/bm25",
        faiss_dir: str = "artifacts/indexes/faiss"
    ) -> None:
        """Loads both BM25 and FAISS index artifacts exported from Colab."""
        self.bm25.load_index(bm25_dir)
        self.dense.load_index(faiss_dir)
        self.is_loaded = True

    def retrieve(
        self,
        query: str,
        top_k: int = 50,
        return_individual: bool = False
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, List[Dict[str, Any]]]]]:
        """Runs BM25 and Dense retrieval and fuses results with RRF.

        Returns:
            fused_results: list of top fused passages
            individual: optional dict with raw bm25 and dense results (useful for feature extraction)
        """
        bm25_res = self.bm25.retrieve(query, top_k=self.top_candidates)
        dense_res = self.dense.retrieve(query, top_k=self.top_candidates)

        fused = self.rrf.fuse(bm25_res, dense_res, top_k=top_k)

        individual = {"bm25": bm25_res, "dense": dense_res} if return_individual else None
        return fused, individual

    def probe_retrieve(self, query: str) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """Lightweight probe retrieval for feature extraction.
        Section V.C: 'For feature extraction, we perform a lightweight probe retrieval (k=2)'.
        """
        fused, individual = self.retrieve(query, top_k=max(self.probe_depth, 5), return_individual=True)
        return fused, individual or {}

