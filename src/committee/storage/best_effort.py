"""BestEffortTraceStore: wraps a TraceStore-shaped backend (Mongo, or any
future best-effort store) so its failures are caught, logged, and swallowed
once — rather than the orchestrator (or each backend individually)
duplicating try/except around every call.

`get_run`/`list_runs` are read paths with no equivalent "never blocks the
debate" requirement (they're not called mid-debate), so failures there
propagate rather than silently returning None/[] — a caller explicitly
asking to read a run deserves to know the read failed, unlike a write that
happens as an unavoidable side effect of a debate in progress.
"""

from __future__ import annotations

import structlog

from committee.models.checkpoint import DebateCheckpoint
from committee.models.trace import DebateTrace, RoundRecord
from committee.storage.trace_store import TraceStore

logger = structlog.get_logger()


class BestEffortTraceStore:
    def __init__(self, inner: TraceStore, backend_name: str):
        self._inner = inner
        self._backend_name = backend_name

    async def save_round(self, run_id: str, round_record: RoundRecord) -> None:
        try:
            await self._inner.save_round(run_id, round_record)
        except Exception as exc:
            logger.warning(
                "best_effort_store_write_failed",
                backend=self._backend_name,
                operation="save_round",
                run_id=run_id,
                error=str(exc),
            )

    async def save_final(self, run_id: str, trace: DebateTrace) -> None:
        try:
            await self._inner.save_final(run_id, trace)
        except Exception as exc:
            logger.warning(
                "best_effort_store_write_failed",
                backend=self._backend_name,
                operation="save_final",
                run_id=run_id,
                error=str(exc),
            )

    async def get_run(self, run_id: str) -> DebateTrace | None:
        return await self._inner.get_run(run_id)

    async def list_runs(self) -> list[str]:
        return await self._inner.list_runs()

    async def save_checkpoint(self, checkpoint: DebateCheckpoint) -> None:
        try:
            await self._inner.save_checkpoint(checkpoint)
        except Exception as exc:
            logger.warning(
                "best_effort_store_write_failed",
                backend=self._backend_name,
                operation="save_checkpoint",
                run_id=checkpoint.run_id,
                error=str(exc),
            )

    async def get_checkpoint(self, run_id: str) -> DebateCheckpoint | None:
        return await self._inner.get_checkpoint(run_id)
