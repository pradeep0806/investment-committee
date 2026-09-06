"""Plug-and-play conflict resolution strategy registration — mirrors
agents/registry.py's pattern. Adding a 4th strategy is one new module + one
line in STRATEGY_MODULES; synthesizer.py never branches on strategy identity.
"""

from __future__ import annotations

from committee.orchestration.conflict_resolution.base import ConflictResolutionStrategy

_REGISTRY: dict[str, type[ConflictResolutionStrategy]] = {}

STRATEGY_MODULES = (
    "committee.orchestration.conflict_resolution.flag_unresolved",
    "committee.orchestration.conflict_resolution.confidence_weighted",
    "committee.orchestration.conflict_resolution.tie_breaker",
)


def register(strategy_id: str):
    def _decorator(cls: type[ConflictResolutionStrategy]) -> type[ConflictResolutionStrategy]:
        _REGISTRY[strategy_id] = cls
        return cls

    return _decorator


def _ensure_strategies_imported() -> None:
    import importlib

    for module_name in STRATEGY_MODULES:
        importlib.import_module(module_name)


def build_strategy(strategy_id: str) -> ConflictResolutionStrategy:
    _ensure_strategies_imported()
    return _REGISTRY[strategy_id]()
