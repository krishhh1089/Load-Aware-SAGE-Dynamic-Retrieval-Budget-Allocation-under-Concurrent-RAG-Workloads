from __future__ import annotations
from typing import List, Dict, Any


def compute_slo_compliance(latencies_ms: List[float], target_slo_ms: float = 5000.0) -> Dict[str, Any]:
    """Calculates SLO compliance rate under target latency deadline T."""
    if not latencies_ms:
        return {
            "target_slo_ms": target_slo_ms,
            "total_queries": 0,
            "compliant_queries": 0,
            "slo_compliance_pct": 100.0
        }

    compliant = sum(1 for lat in latencies_ms if lat <= target_slo_ms)
    total = len(latencies_ms)
    compliance_pct = (compliant / total) * 100.0

    return {
        "target_slo_ms": float(target_slo_ms),
        "total_queries": total,
        "compliant_queries": compliant,
        "violations": total - compliant,
        "slo_compliance_pct": float(round(compliance_pct, 2))
    }

