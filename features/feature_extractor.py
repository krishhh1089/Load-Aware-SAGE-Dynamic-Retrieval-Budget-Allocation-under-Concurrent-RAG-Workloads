from __future__ import annotations
import os
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from features.query_features import QueryFeatureExtractor
from features.retrieval_features import RetrievalFeatureExtractor


class SAGEFeatureExtractor:
    """Master feature extractor for SAGE inference.
    Enforces exact feature ordering and schema validation matching Colab exports.
    """

    DEFAULT_FEATURE_NAMES = [
        "query_length_chars",
        "query_length_tokens",
        "query_has_question_mark",
        "query_avg_word_len",
        "top1_score",
        "score_drop",
        "score_mean",
        "score_std",
        "sparse_dense_agreement",
        "target_slo_ms"
    ]

    def __init__(self, schema_path: str = "artifacts/metadata/feature_schema.json"):
        self.schema_path = schema_path
        self.feature_names = list(self.DEFAULT_FEATURE_NAMES)
        self.load_schema()

    def load_schema(self) -> None:
        if os.path.exists(self.schema_path):
            try:
                with open(self.schema_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "feature_names" in data and isinstance(data["feature_names"], list):
                        self.feature_names = data["feature_names"]
            except Exception as e:
                print(f"Warning: Failed to load schema from {self.schema_path} ({e}). Using default schema.")

    def extract_features(
        self,
        query: str,
        fused_candidates: List[Dict[str, Any]],
        bm25_candidates: Optional[List[Dict[str, Any]]] = None,
        dense_candidates: Optional[List[Dict[str, Any]]] = None,
        target_slo_ms: float = 5000.0
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Extracts features and returns both a 2D array (1, n_features) and feature dict."""
        q_features = QueryFeatureExtractor.extract(query)
        r_features = RetrievalFeatureExtractor.extract(
            fused_candidates=fused_candidates,
            bm25_candidates=bm25_candidates,
            dense_candidates=dense_candidates
        )

        raw_dict = {
            **q_features,
            **r_features,
            "target_slo_ms": float(target_slo_ms)
        }

        # Align with exact schema ordering
        feature_vector = []
        for name in self.feature_names:
            val = raw_dict.get(name, 0.0)
            feature_vector.append(float(val))

        vec_np = np.array(feature_vector, dtype=np.float64).reshape(1, -1)
        ordered_dict = {name: float(val) for name, val in zip(self.feature_names, feature_vector)}

        return vec_np, ordered_dict

