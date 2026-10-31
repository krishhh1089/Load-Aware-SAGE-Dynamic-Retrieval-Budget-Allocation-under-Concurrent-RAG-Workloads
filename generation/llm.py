from __future__ import annotations
import time
from typing import Dict, Any, List, Optional
from generation.openai_client import OpenAIClient


DEFAULT_SYSTEM_PROMPT = (
    "You are an accurate, concise question answering assistant. "
    "Answer the user query faithfully based strictly on the provided context passages. "
    "If the answer cannot be determined from the context, state 'I cannot determine the answer from the provided context.'."
)


class MockLLMClient:
    """Mock LLM generator for local development and reproducible testing without API credentials."""

    def __init__(self, base_latency_ms: float = 35.0, latency_per_token_ms: float = 0.2):
        self.base_latency_ms = base_latency_ms
        self.latency_per_token_ms = latency_per_token_ms

    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        start = time.perf_counter()

        # Extract context passages if present to produce grounded mock answer
        context_lines = [line.strip() for line in user_prompt.split("\n") if line.strip() and not line.startswith("User Query:")]
        answer = "Based on the retrieved context: " + (context_lines[1] if len(context_lines) > 1 else "grounded answer derived from retrieved passages.")

        # Scaled sleep for lightweight simulation without slowing test execution
        simulated_ms = self.base_latency_ms + len(answer.split()) * self.latency_per_token_ms
        time.sleep(min(0.05, simulated_ms / 1000.0))

        latency_ms = (time.perf_counter() - start) * 1000.0
        prompt_tokens = int(len(user_prompt.split()) * 1.3)
        completion_tokens = int(len(answer.split()) * 1.3)

        return {
            "answer": answer,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "model": "mock-llm-v1"
        }


class SAGEGenerationManager:
    """Dispatches generation requests to configured LLM provider (OpenAI or Mock)
    and tracks generation metrics.
    """

    def __init__(
        self,
        provider: str = "mock",
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 150,
        api_key: Optional[str] = None
    ):
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if self.provider == "openai":
            self.client = OpenAIClient(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            self.client = MockLLMClient()

    def generate_response(
        self,
        query: str,
        context_string: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        sys_p = system_prompt or DEFAULT_SYSTEM_PROMPT
        user_p = (
            f"Context passages:\n{context_string}\n\n"
            f"User Query: {query}\n"
            f"Answer:"
        )
        return self.client.generate(system_prompt=sys_p, user_prompt=user_p)

