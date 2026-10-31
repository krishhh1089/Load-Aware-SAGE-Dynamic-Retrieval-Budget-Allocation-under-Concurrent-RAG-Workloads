from __future__ import annotations
import os
import time
from typing import Dict, Any, Optional, Tuple


class OpenAIClient:
    """Production client for OpenAI LLMs with credential management and latency tracking."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 150,
        timeout: float = 10.0
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = None

    def initialize(self) -> None:
        if self._client is not None:
            return
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment or configuration. "
                "Set OPENAI_API_KEY in your .env file or use mock generation mode."
            )
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, timeout=self.timeout)
        except ImportError:
            raise ImportError("openai package is required. Install via 'pip install openai'.")

    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Calls OpenAI Chat Completion API, records latency and token counts."""
        self.initialize()

        start_time = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        content = response.choices[0].message.content or ""
        usage = response.usage

        return {
            "answer": content.strip(),
            "latency_ms": latency_ms,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "model": self.model
        }

