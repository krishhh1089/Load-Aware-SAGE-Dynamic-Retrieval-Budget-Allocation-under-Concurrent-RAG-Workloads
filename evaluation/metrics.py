from __future__ import annotations
from typing import List, Dict, Any
import numpy as np

from evaluation.exact_match import compute_exact_match
from evaluation.recall import compute_recall_at_k
from evaluation.latency import compute_latency_percentiles
from evaluation.slo import compute_slo_compliance


class SAGEEvaluator:
    """Evaluates SAGE runtime across quality, latency, cost, and SLO metrics matching Table I of the paper."""

    def __init__(self, target_slo_ms: float = 5000.0, static_baseline_k: int = 20):
        self.target_slo_ms = target_slo_ms
        self.static_baseline_k = static_baseline_k

    def evaluate_batch(self, query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates comprehensive benchmark metrics over a batch of query results.

        Each item in query_results should contain:
            - 'predicted_k': int
            - 'latency_ms': float
            - 'prediction': str
            - 'ground_truth': str or list of str (optional)
            - 'retrieved_passages': list of dicts (optional)
        """
        if not query_results:
            return {"error": "No query results provided."}

        budgets = [res.get("predicted_k", 0) for res in query_results]
        latencies = [res.get("latency_ms", 0.0) for res in query_results]

        # EM and Recall calculations where ground truths are present
        em_scores = []
        recall_scores = []
        for res in query_results:
            gt = res.get("ground_truth")
            pred = res.get("prediction", "")
            passages = res.get("retrieved_passages", [])

            if gt:
                em_scores.append(compute_exact_match(pred, gt))
                if passages:
                    recall_scores.append(compute_recall_at_k(passages, gt, k=20))

        avg_k = float(np.mean(budgets)) if budgets else 0.0
        relative_cost_pct = (avg_k / self.static_baseline_k) * 100.0

        latency_stats = compute_latency_percentiles(latencies)
        slo_stats = compute_slo_compliance(latencies, self.target_slo_ms)

        summary = {
            "total_queries": len(query_results),
            "avg_k": float(round(avg_k, 2)),
            "relative_cost_pct": float(round(relative_cost_pct, 1)),
            "slo_compliance_pct": slo_stats["slo_compliance_pct"],
            "p95_latency_ms": latency_stats["p95"],
            "p50_latency_ms": latency_stats["p50"],
            "mean_latency_ms": latency_stats["mean"],
            "exact_match_pct": float(round(np.mean(em_scores) * 100.0, 2)) if em_scores else None,
            "recall_at_20_pct": float(round(np.mean(recall_scores) * 100.0, 2)) if recall_scores else None
        }
        return summary

