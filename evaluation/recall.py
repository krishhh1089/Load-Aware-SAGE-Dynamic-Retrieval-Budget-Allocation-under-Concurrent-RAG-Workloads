from __future__ import annotations
from typing import List, Dict, Any, Set, Union


def compute_recall_at_k(
    retrieved_passages: List[Dict[str, Any]],
    ground_truth_ids_or_answers: Union[List[str], Set[str]],
    k: int = 20
) -> float:
    """Computes Recall@k.
    Returns 1.0 if any ground truth passage or answer substring is within top k passages, else 0.0.
    """
    if not ground_truth_ids_or_answers:
        return 0.0

    top_k = retrieved_passages[:k]
    gt_set = set(ground_truth_ids_or_answers)

    # Check ID match
    retrieved_ids = {p.get("id") for p in top_k if p.get("id")}
    if retrieved_ids & gt_set:
        return 1.0

    # Check answer text inclusion in passage text
    for p in top_k:
        p_text = p.get("text", "").lower()
        for gt in ground_truth_ids_or_answers:
            if str(gt).lower() in p_text:
                return 1.0

    return 0.0

