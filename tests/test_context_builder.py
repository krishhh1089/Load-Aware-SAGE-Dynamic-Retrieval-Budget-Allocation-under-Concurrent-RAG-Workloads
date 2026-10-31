import pytest
from generation.context_builder import ContextBuilder


def test_context_builder_budget_selection():
    builder = ContextBuilder(max_context_tokens=1000)
    passages = [
        {"id": f"doc_{i}", "text": f"This is passage content number {i}.", "title": f"Title {i}"}
        for i in range(10)
    ]

    context_str, selected, token_est = builder.build_context(passages, budget_k=3)
    assert len(selected) == 3
    assert "[1] (Document: doc_0)" in context_str
    assert "[3] (Document: doc_2)" in context_str
    assert "doc_3" not in context_str


def test_context_builder_deduplication():
    builder = ContextBuilder(max_context_tokens=1000)
    passages = [
        {"id": "doc_1", "text": "Same repeated text."},
        {"id": "doc_1", "text": "Same repeated text."}, # Duplicate ID
        {"id": "doc_2", "text": "Same repeated text."}, # Duplicate text
        {"id": "doc_3", "text": "Unique text content."}
    ]

    context_str, selected, _ = builder.build_context(passages, budget_k=4)
    assert len(selected) == 2
    assert selected[0]["id"] == "doc_1"
    assert selected[1]["id"] == "doc_3"

