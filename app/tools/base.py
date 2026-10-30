"""Tool abstraction and registry.

A Tool is a plain Python function plus a JSON schema the LLM sees. The registry
executes tool calls requested by an agent and *never raises*: any failure is
returned to the agent as {"error": ...} so the model can reason about it
(retry with fixed arguments, or report the gap) instead of crashing the run.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger(__name__)


class ToolError(Exception):
    """A domain error a tool wants to report to the agent (e.g. unknown applicant)."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema for the arguments
    func: Callable[..., dict]

    def openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self, names: list[str]) -> list[dict]:
        return [self._tools[n].openai_schema() for n in names]

    def execute(self, name: str, arguments: str | dict, allowed: list[str] | None = None) -> dict[str, Any]:
        start = time.perf_counter()
        if allowed is not None and name not in allowed:
            return {"error": f"Tool '{name}' is not available to this agent. Allowed: {allowed}"}
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool '{name}'."}
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
        except json.JSONDecodeError as exc:
            return {"error": f"Arguments were not valid JSON: {exc}"}
        try:
            result = tool.func(**args)
            log.info("tool=%s args=%s ok in %.1f ms", name, args, (time.perf_counter() - start) * 1000)
            return result
        except ToolError as exc:
            log.warning("tool=%s args=%s domain error: %s", name, args, exc)
            return {"error": str(exc)}
        except TypeError as exc:  # wrong / missing argument names
            log.warning("tool=%s bad arguments %s: %s", name, args, exc)
            return {"error": f"Invalid arguments for {name}: {exc}"}
        except Exception as exc:  # noqa: BLE001 - last line of defence
            log.exception("tool=%s crashed", name)
            return {"error": f"Tool {name} failed unexpectedly: {type(exc).__name__}"}
