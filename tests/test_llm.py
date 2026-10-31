import pytest
from generation.llm import SAGEGenerationManager, MockLLMClient


def test_mock_llm_generation():
    client = MockLLMClient(base_latency_ms=10.0)
    result = client.generate(
        system_prompt="System instructions.",
        user_prompt="Context passages:\n[1] Paris is France capital.\nUser Query: Capital of France?"
    )

    assert "answer" in result
    assert "latency_ms" in result
    assert result["latency_ms"] > 0
    assert result["prompt_tokens"] > 0
    assert result["completion_tokens"] > 0


def test_generation_manager_dispatch():
    manager = SAGEGenerationManager(provider="mock")
    res = manager.generate_response(
        query="What is SAGE?",
        context_string="[1] SAGE is an adaptive retrieval policy."
    )
    assert res["model"] == "mock-llm-v1"
    assert len(res["answer"]) > 0

