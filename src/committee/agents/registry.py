"""Plug-and-play agent registration.

Adding a 5th agent lens never touches this file's logic — it just imports one
more module in `AGENT_MODULES` below so that module's `@register(...)`
decorator runs and populates `_REGISTRY`.
"""

from __future__ import annotations

from committee.agents.base import AnalystAgent
from committee.orchestration.budget_gate import BudgetGate

_REGISTRY: dict[str, type[AnalystAgent]] = {}

# Import order also fixes the deterministic agent ordering used by round-robin
# directives (e.g. explore mode's rotating "argue against majority" prompt).
# Adding a 5th lens is exactly "one new module + one line here."
AGENT_MODULES = (
    "committee.agents.fundamentals",
    "committee.agents.market_sentiment",
    "committee.agents.risk_contrarian",
    "committee.agents.macro_context",
)

DEFAULT_AGENT_ROLES = (
    "fundamentals",
    "market_sentiment",
    "risk_contrarian",
    "macro_context",
)


def register(agent_id: str):
    def _decorator(cls: type[AnalystAgent]) -> type[AnalystAgent]:
        _REGISTRY[agent_id] = cls
        return cls

    return _decorator


def _ensure_agents_imported() -> None:
    import importlib

    for module_name in AGENT_MODULES:
        importlib.import_module(module_name)


def build_agents(budget_gate: BudgetGate, agent_roles: list[str] | None = None) -> list[AnalystAgent]:
    _ensure_agents_imported()
    roles = agent_roles or list(DEFAULT_AGENT_ROLES)
    return [_REGISTRY[role](budget_gate=budget_gate) for role in roles]
