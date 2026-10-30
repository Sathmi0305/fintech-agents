from app.tools import affordability, credit_bureau, transactions
from app.tools.base import Tool, ToolError, ToolRegistry


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for module in (credit_bureau, transactions, affordability):
        registry.register(module.TOOL)
    return registry


__all__ = ["Tool", "ToolError", "ToolRegistry", "build_registry"]
