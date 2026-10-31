import pytest
from retrieval.rrf import ReciprocalRankFusion


def test_rrf_scoring():
    rrf = ReciprocalRankFusion(k=60)
    bm25_res = [
        {"id": "docA", "text": "A", "score": 10.0},
        {"id": "docB", "text": "B", "score": 5.0}
    ]
    dense_res = [
        {"id": "docB", "text": "B", "score": 0.95},
        {"id": "docC", "text": "C", "score": 0.80}
    ]

    fused = rrf.fuse(bm25_res, dense_res)
    assert len(fused) == 3

    # docB appears in both: rank 2 in bm25, rank 1 in dense
    # score(docB) = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ~ 0.016129 + 0.016393 = 0.032522
    # score(docA) = 1/(60+1) = 1/61 ~ 0.016393
    # docB should rank #1!
    assert fused[0]["id"] == "docB"
    assert fused[0]["rank"] == 1
    assert fused[0]["score"] > fused[1]["score"]


def test_rrf_top_k_limiting():
    rrf = ReciprocalRankFusion(k=60)
    bm25_res = [{"id": f"d_{i}", "text": f"text {i}"} for i in range(10)]
    dense_res = [{"id": f"d_{i}", "text": f"text {i}"} for i in range(5, 15)]

    fused = rrf.fuse(bm25_res, dense_res, top_k=5)
    assert len(fused) == 5

