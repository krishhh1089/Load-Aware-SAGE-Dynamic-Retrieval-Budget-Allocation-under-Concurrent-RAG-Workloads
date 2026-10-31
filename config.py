from __future__ import annotations
import os
from typing import Dict, Any, List, Optional
import yaml
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


class Config:
    """Central configuration loader for SAGE runtime."""

    def __init__(self, config_dict: Dict[str, Any]):
        self._raw = config_dict
        self.artifacts = config_dict.get("artifacts", {})
        self.retrieval = config_dict.get("retrieval", {})
        self.policy = config_dict.get("policy", {})
        self.generation = config_dict.get("generation", {})
        self.slo = config_dict.get("slo", {})
        self.logging = config_dict.get("logging", {})

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> Config:
        path = config_path or os.getenv("SAGE_CONFIG_PATH", "configs/config.yaml")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Allow environment variable overrides
        if os.getenv("OPENAI_API_KEY"):
            data.setdefault("generation", {})["openai_api_key"] = os.getenv("OPENAI_API_KEY")

        return cls(data)

    def get(self, key: str, default: Any = None) -> Any:
        return self._raw.get(key, default)


# Global default configuration instance
def get_config(config_path: Optional[str] = None) -> Config:
    return Config.load(config_path)

