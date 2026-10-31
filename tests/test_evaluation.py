import pytest
from evaluation.exact_match import compute_exact_match, normalize_answer
from evaluation.recall import compute_recall_at_k
from evaluation.latency import compute_latency_percentiles
from evaluation.slo import compute_slo_compliance
from evaluation.metrics import SAGEEvaluator


def test_normalization_and_exact_match():
    pred = "Paris"
    gt = "paris"
    assert compute_exact_match(pred, gt) == 1.0

    pred2 = "The capital is Paris."
    assert compute_exact_match(pred2, ["Paris"]) == 1.0

    pred3 = "Rome"
    assert compute_exact_match(pred3, ["Paris"]) == 0.0


def test_recall_at_k():
    passages = [
        {"id": "doc1", "text": "Washington D.C. is the capital."},
        {"id": "doc2", "text": "London is the capital of England."}
    ]
    # Match by ID
    assert compute_recall_at_k(passages, ["doc1"], k=1) == 1.0
    assert compute_recall_at_k(passages, ["doc2"], k=1) == 0.0
    assert compute_recall_at_k(passages, ["doc2"], k=2) == 1.0

    # Match by text
    assert compute_recall_at_k(passages, ["Washington"], k=1) == 1.0


def test_latency_and_slo():
    lats = [100.0, 200.0, 300.0, 400.0, 6000.0]
    percentiles = compute_latency_percentiles(lats)
    assert percentiles["p50"] == 300.0

    slo = compute_slo_compliance(lats, target_slo_ms=5000.0)
    assert slo["compliant_queries"] == 4
    assert slo["total_queries"] == 5
    assert slo["slo_compliance_pct"] == 80.0


def test_sage_batch_evaluator():
    evaluator = SAGEEvaluator(target_slo_ms=5000.0, static_baseline_k=20)
    batch = [
        {
            "predicted_k": 5,
            "latency_ms": 120.0,
            "prediction": "Paris",
            "ground_truth": "Paris",
            "retrieved_passages": [{"id": "doc1", "text": "Paris is capital."}]
        },
        {
            "predicted_k": 10,
            "latency_ms": 250.0,
            "prediction": "Berlin",
            "ground_truth": "Berlin",
            "retrieved_passages": [{"id": "doc2", "text": "Berlin is capital."}]
        }
    ]
    res = evaluator.evaluate_batch(batch)
    assert res["total_queries"] == 2
    assert res["avg_k"] == 7.5
    assert res["relative_cost_pct"] == 37.5 # (7.5 / 20) * 100%
    assert res["exact_match_pct"] == 100.0
    assert res["slo_compliance_pct"] == 100.0

