"""The agent loop: Reason -> Act (tool call) -> Observe (tool result) -> repeat -> Answer.

Each agent has a role prompt, an allow-list of tools, and a step budget. The
loop is the same for every agent; only the prompt and tools differ.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from app.llm import LLMClient, LLMError
from app.tools.base import ToolRegistry

log = logging.getLogger(__name__)

EventCallback = Callable[[str, dict], None]


class AgentError(Exception):
    """The agent could not produce a usable answer."""


@dataclass
class AgentResult:
    agent: str
    output: dict
    trace: list[dict] = field(default_factory=list)  # every tool call and its result, in order
    llm_calls: int = 0


def extract_json(text: str | None) -> dict | None:
    """Pull a JSON object out of a model reply (tolerates ```json fences and stray prose)."""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    for candidate in (cleaned, text[text.find("{"): text.rfind("}") + 1] if "{" in text else ""):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None


class BaseAgent:
    name = "agent"
    system_prompt = ""
    tool_names: list[str] = []
    required_keys: list[str] = []

    def __init__(self, llm: LLMClient, registry: ToolRegistry, max_steps: int = 6,
                 on_event: EventCallback | None = None) -> None:
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self.on_event = on_event or (lambda kind, payload: None)

    def _emit(self, kind: str, **payload: Any) -> None:
        self.on_event(kind, {"agent": self.name, **payload})

    def run(self, task: str) -> AgentResult:
        messages: list[dict] = [{"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": task}]
        tools = self.registry.schemas(self.tool_names)
        result = AgentResult(agent=self.name, output={})
        format_retry_used = False
        self._emit("agent_start", task=task)

        for step in range(1, self.max_steps + 1):
            try:
                response = self.llm.chat(messages, tools)
            except LLMError:
                log.exception("%s: LLM call failed at step %d", self.name, step)
                raise
            result.llm_calls += 1

            # ACT: the model asked for tools. Execute them and feed results back (OBSERVE).
            if response.tool_calls:
                if response.content:
                    self._emit("thought", text=response.content)
                messages.append({
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [{"id": c.id, "type": "function",
                                    "function": {"name": c.name, "arguments": c.arguments}}
                                   for c in response.tool_calls],
                })
                for call in response.tool_calls:
                    self._emit("tool_call", tool=call.name, arguments=call.arguments)
                    output = self.registry.execute(call.name, call.arguments, allowed=self.tool_names)
                    self._emit("tool_result", tool=call.name, result=output)
                    result.trace.append({"step": step, "tool": call.name,
                                         "arguments": call.arguments, "result": output})
                    messages.append({"role": "tool", "tool_call_id": call.id,
                                     "content": json.dumps(output, default=str)})
                continue

            # ANSWER: no more tool calls; the reply should be the agent's structured output.
            parsed = extract_json(response.content)
            missing = [k for k in self.required_keys if parsed is not None and k not in parsed]
            if parsed is not None and not missing:
                result.output = parsed
                self._emit("agent_done", output=parsed, llm_calls=result.llm_calls)
                log.info("%s finished in %d LLM calls, %d tool calls", self.name, result.llm_calls, len(result.trace))
                return result

            if format_retry_used:
                raise AgentError(f"{self.name} returned an invalid final answer twice: {response.content!r:.300}")
            format_retry_used = True
            problem = f"missing keys {missing}" if parsed is not None else "not valid JSON"
            log.warning("%s: final answer %s; asking once more", self.name, problem)
            messages.append({"role": "assistant", "content": response.content or ""})
            messages.append({"role": "user", "content":
                             f"Your answer was {problem}. Reply with ONLY the JSON object in the required schema."})

        raise AgentError(f"{self.name} hit its step budget ({self.max_steps}) without a final answer.")
