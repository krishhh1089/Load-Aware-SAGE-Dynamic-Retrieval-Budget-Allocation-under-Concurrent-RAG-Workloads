from __future__ import annotations
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds LLM context from ranked retrieval results according to SAGE policy budget k*."""

    def __init__(self, max_context_tokens: int = 2048, chars_per_token_approx: float = 4.0):
        self.max_context_tokens = max_context_tokens
        self.max_context_chars = int(max_context_tokens * chars_per_token_approx)

    def build_context(
        self,
        ranked_passages: List[Dict[str, Any]],
        budget_k: int
    ) -> Tuple[str, List[Dict[str, Any]], int]:
        """Selects top budget_k non-duplicate passages and constructs prompt context.

        Returns:
            context_string: formatted context with passage references
            selected_passages: list of deduplicated passages included
            token_count_estimate: estimated tokens used
        """
        seen_ids = set()
        seen_texts = set()
        selected: List[Dict[str, Any]] = []

        # 1. Deduplicate and take top budget_k
        for passage in ranked_passages:
            if len(selected) >= budget_k:
                break

            doc_id = passage.get("id")
            raw_text = passage.get("text", "").strip()

            if not raw_text:
                continue

            # Deduplication by ID or normalized text
            text_snippet = raw_text[:120].lower()
            if (doc_id and doc_id in seen_ids) or (text_snippet in seen_texts):
                continue

            if doc_id:
                seen_ids.add(doc_id)
            seen_texts.add(text_snippet)
            selected.append(passage)

        # 2. Build context string respecting max length
        formatted_chunks = []
        current_chars = 0

        for rank, p in enumerate(selected, start=1):
            doc_id = p.get("id", f"doc_{rank}")
            title = p.get("title", "")
            title_prefix = f"Title: {title} | " if title else ""
            header = f"[{rank}] (Document: {doc_id}) {title_prefix}"
            body = p.get("text", "").strip()
            chunk_str = f"{header}\n{body}\n"

            if current_chars + len(chunk_str) > self.max_context_chars and formatted_chunks:
                logger.warning(
                    f"Context truncated at passage {rank}/{len(selected)} to respect token limit."
                )
                break

            formatted_chunks.append(chunk_str)
            current_chars += len(chunk_str)

        context_string = "\n".join(formatted_chunks) if formatted_chunks else "No relevant context available."
        estimated_tokens = int(current_chars / 4.0)

        logger.info(
            f"Context built: selected {len(selected)} passages (budget k={budget_k}), "
            f"approx tokens: {estimated_tokens}"
        )

        return context_string, selected, estimated_tokens

