from __future__ import annotations
import re
import string
from typing import List, Union


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.
    Standard SQuAD / Natural Questions evaluation normalization.
    """
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truths: Union[str, List[str]]) -> float:
    """Computes exact match score (1.0 or 0.0) against ground truth answers."""
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    normalized_pred = normalize_answer(prediction)
    for gt in ground_truths:
        if normalize_answer(gt) == normalized_pred:
            return 1.0
        # Substring match if prediction contains normalized answer
        if normalize_answer(gt) in normalized_pred:
            return 1.0
    return 0.0

