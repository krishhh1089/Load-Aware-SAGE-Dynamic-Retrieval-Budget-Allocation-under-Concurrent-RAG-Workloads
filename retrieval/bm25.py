from __future__ import annotations
import os
import re
import math
import json
import joblib
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple


class ArtifactNotFoundError(FileNotFoundError):
    """Raised when a required artifact exported from Google Colab is missing."""
    pass


def default_tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())


class BM25Retriever:
    """Production BM25 runtime retriever.
    Loads pre-indexed BM25 artifacts exported from Google Colab and runs inference queries.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.passages: List[Dict[str, Any]] = []
        self.doc_len: List[int] = []
        self.avg_dl: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.term_freqs: List[Dict[str, int]] = []
        self.n_docs: int = 0
        self.is_loaded: bool = False

    def load_index(self, index_dir: str = "artifacts/indexes/bm25") -> None:
        """Loads BM25 index artifact exported from Google Colab."""
        if not os.path.exists(index_dir):
            raise ArtifactNotFoundError(
                f"BM25 index directory not found at: '{index_dir}'. "
                f"Please export the BM25 index from Google Colab into this directory."
            )

        # Support both single joblib file and directory structures
        primary_artifact = os.path.join(index_dir, "bm25_index.joblib")
        if not os.path.exists(primary_artifact):
            pkl_cand = os.path.join(index_dir, "bm25_index.pkl")
            if os.path.exists(pkl_cand):
                primary_artifact = pkl_cand
            else:
                files = os.listdir(index_dir)
                raise ArtifactNotFoundError(
                    f"No valid BM25 index file found in '{index_dir}' (found: {files}). "
                    f"Expected 'bm25_index.joblib' exported from Colab."
                )

        data = joblib.load(primary_artifact)
        self.k1 = data.get("k1", self.k1)
        self.b = data.get("b", self.b)
        self.passages = data.get("passages", [])
        self.doc_len = data.get("doc_len", [])
        self.avg_dl = data.get("avg_dl", 0.0)
        self.doc_freqs = data.get("doc_freqs", {})
        self.idf = data.get("idf", {})
        self.term_freqs = data.get("term_freqs", [])
        self.n_docs = len(self.passages)
        self.is_loaded = True

    def retrieve(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """Retrieve top_k passages using BM25 ranking algorithm."""
        if not self.is_loaded or self.n_docs == 0:
            raise RuntimeError("BM25 index is not loaded. Call load_index() before retrieve().")

        tokens = default_tokenize(query)
        if not tokens:
            return []

        scores = [0.0] * self.n_docs
        for token in tokens:
            if token not in self.idf:
                continue
            token_idf = self.idf[token]
            for doc_idx in range(self.n_docs):
                tf = self.term_freqs[doc_idx].get(token, 0)
                if tf > 0:
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_len[doc_idx] / max(1e-5, self.avg_dl)))
                    scores[doc_idx] += token_idf * ((tf * (self.k1 + 1.0)) / denom)

        # Get top-k indices
        ranked_indices = sorted(range(self.n_docs), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            passage = self.passages[idx]
            results.append({
                "id": passage.get("id", f"doc_{idx}"),
                "text": passage.get("text", passage.get("content", "")),
                "title": passage.get("title", ""),
                "score": float(scores[idx]),
                "rank": rank,
                "retriever": "bm25",
                "metadata": passage.get("metadata", {})
            })
        return results

    @classmethod
    def save_index(cls, instance: BM25Retriever, output_dir: str) -> str:
        """Utility used by export scripts or testing fixtures to serialize index."""
        os.makedirs(output_dir, exist_ok=True)
        target = os.path.join(output_dir, "bm25_index.joblib")
        joblib.dump({
            "k1": instance.k1,
            "b": instance.b,
            "passages": instance.passages,
            "doc_len": instance.doc_len,
            "avg_dl": instance.avg_dl,
            "doc_freqs": instance.doc_freqs,
            "idf": instance.idf,
            "term_freqs": instance.term_freqs,
            "n_docs": instance.n_docs
        }, target)
        return target

