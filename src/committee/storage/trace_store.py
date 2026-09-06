"""TraceStore: protocol for persisting a debate's trace (CLAUDE.md §1.3, §5
step 7). JSON is the only implementation required to succeed — it's the
literal Core requirement ("full debate trace saved as structured JSON") and
the source of truth, independent of whether Mongo is reachable. Mongo is
best-effort (see BestEffortTraceStore) and never blocks the debate.

Redis is deliberately NOT behind this protocol — it's a pub/sub channel +
hot cache, a different shape of concern entirely (see redis_bus.py).
"""

from __future__ import annotations

from typing import Protocol

from committee.models.trace import DebateTrace, RoundRecord


class TraceStore(Protocol):
    async def save_round(self, run_id: str, round_record: RoundRecord) -> None: ...

    async def save_final(self, run_id: str, trace: DebateTrace) -> None: ...

    async def get_run(self, run_id: str) -> DebateTrace | None: ...

    async def list_runs(self) -> list[str]: ...
