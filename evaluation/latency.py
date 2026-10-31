from __future__ import annotations
from typing import List, Dict, Any
import numpy as np


def compute_latency_percentiles(latencies_ms: List[float]) -> Dict[str, float]:
    """Computes mean, median (p50), p90, p95, and p99 latencies."""
    if not latencies_ms:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}

    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "mean": float(round(np.mean(arr), 2)),
        "p50": float(round(np.percentile(arr, 50), 2)),
        "p90": float(round(np.percentile(arr, 90), 2)),
        "p95": float(round(np.percentile(arr, 95), 2)),
        "p99": float(round(np.percentile(arr, 99), 2))
    }

