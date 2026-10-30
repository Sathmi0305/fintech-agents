"""Thin LLM wrapper.

Agents talk to `LLMClient.chat()` and get back a provider-neutral `LLMResponse`.
This keeps the agent loop independent of the OpenAI SDK and lets us swap in the
offline MockLLMClient for demos and tests.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

from app.config import Settings

log = logging.getLogger(__name__)


class LLMError(Exception):
    """The model could not be reached or returned an unusable response."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON string, exactly as the model produced it


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class LLMClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse: ...


class OpenAIClient:
    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI  # imported lazily so --mock works without the package configured

        self._settings = settings
        # The SDK retries connection errors, 429s and 5xx with exponential backoff.
        self._client = OpenAI(api_key=settings.gemini_api_key, timeout=settings.request_timeout_s, max_retries=3)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        import openai

        kwargs: dict = {"model": self._settings.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self._settings.temperature is not None:
            kwargs["temperature"] = self._settings.temperature

        start = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except openai.AuthenticationError as exc:
            raise LLMError("OpenAI rejected the API key (401). Check GEMINI_API_KEY.") from exc
        except openai.NotFoundError as exc:
            raise LLMError(f"Model '{self._settings.model}' not found. Set OPENAI_MODEL to a model you can access.") from exc
        except openai.RateLimitError as exc:
            raise LLMError("OpenAI rate limit or quota exceeded after retries.") from exc
        except openai.APIConnectionError as exc:
            raise LLMError("Could not connect to the OpenAI API (network/timeout).") from exc
        except openai.BadRequestError as exc:
            raise LLMError(f"OpenAI rejected the request: {exc.message}") from exc
        except openai.APIError as exc:
            raise LLMError(f"OpenAI API error: {exc}") from exc

        msg = resp.choices[0].message
        usage = resp.usage.model_dump() if resp.usage else {}
        log.info("llm model=%s %.0f ms tokens=%s", self._settings.model, (time.perf_counter() - start) * 1000,
                 usage.get("total_tokens"))
        calls = [ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
                 for tc in (msg.tool_calls or []) if tc.type == "function"]
        return LLMResponse(content=msg.content, tool_calls=calls, usage=usage)


def build_llm(settings: Settings) -> LLMClient:
    if settings.use_mock_llm:
        from app.mock_llm import MockLLMClient

        return MockLLMClient()
    return OpenAIClient(settings)
