"""Application settings, loaded from environment variables / a .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when the configuration is invalid (e.g. missing API key)."""


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None
    model: str
    temperature: float | None
    request_timeout_s: float
    max_agent_steps: int      # max LLM round-trips per agent run (stops runaway loops)
    max_revisions: int        # max times the Underwriter may send questions back to the Analyst
    log_level: str
    use_mock_llm: bool        # offline demo mode: deterministic rule-based "LLM", no API calls


def _optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def load_settings(use_mock_llm: bool | None = None) -> Settings:
    load_dotenv()
    mock = use_mock_llm if use_mock_llm is not None else os.getenv("USE_MOCK_LLM", "false").lower() == "true"
    settings = Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=_optional_float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
        request_timeout_s=float(os.getenv("OPENAI_TIMEOUT_S", "60")),
        max_agent_steps=int(os.getenv("MAX_AGENT_STEPS", "6")),
        max_revisions=int(os.getenv("MAX_REVISIONS", "1")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        use_mock_llm=mock,
    )
    if not settings.use_mock_llm and not settings.gemini_api_key:
        raise ConfigError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or run with --mock for the offline demo."
        )
    return settings
