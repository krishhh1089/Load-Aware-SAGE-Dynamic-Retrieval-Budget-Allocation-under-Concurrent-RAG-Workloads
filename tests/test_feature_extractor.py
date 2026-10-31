import pytest
import numpy as np
from features.query_features import QueryFeatureExtractor
from features.retrieval_features import RetrievalFeatureExtractor
from features.feature_extractor import SAGEFeatureExtractor


def test_query_features():
    q = "What is the capital of France?"
    feats = QueryFeatureExtractor.extract(q)
    assert feats["query_has_question_mark"] == 1.0
    assert feats["query_length_chars"] == len(q)
    assert feats["query_length_tokens"] == 6
    assert feats["query_avg_word_len"] > 0


def test_retrieval_features():
    candidates = [
        {"id": "doc1", "score": 0.035},
        {"id": "doc2", "score": 0.025},
        {"id": "doc3", "score": 0.015}
    ]
    bm25 = [{"id": "doc1"}, {"id": "doc2"}]
    dense = [{"id": "doc2"}, {"id": "doc3"}]

    r_feats = RetrievalFeatureExtractor.extract(candidates, bm25, dense)
    assert r_feats["top1_score"] == 0.035
    assert abs(r_feats["score_drop"] - 0.01) < 1e-5
    assert r_feats["sparse_dense_agreement"] > 0.0


def test_sage_master_feature_extractor(tmp_path):
    extractor = SAGEFeatureExtractor()
    query = "Where was Alan Turing born?"
    fused = [{"id": "d1", "score": 0.03}, {"id": "d2", "score": 0.02}]

    vec, feat_dict = extractor.extract_features(
        query=query,
        fused_candidates=fused,
        target_slo_ms=5000.0
    )

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (1, len(extractor.feature_names))
    assert feat_dict["target_slo_ms"] == 5000.0
    assert feat_dict["query_has_question_mark"] == 1.0

