from __future__ import annotations
import re
from typing import Dict, Any


class QueryFeatureExtractor:
    """Extracts query lexical and structural features according to SAGE paper specifications."""

    @classmethod
    def extract(cls, query: str) -> Dict[str, float]:
        clean_query = query.strip()
        tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", clean_query)
        char_len = len(clean_query)
        token_count = len(tokens)

        has_question_mark = 1.0 if "?" in query else 0.0
        avg_word_len = (sum(len(t) for t in tokens) / max(1, token_count)) if token_count > 0 else 0.0

        return {
            "query_length_chars": float(char_len),
            "query_length_tokens": float(token_count),
            "query_has_question_mark": float(has_question_mark),
            "query_avg_word_len": float(round(avg_word_len, 4))
        }

