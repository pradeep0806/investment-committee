"""JsonStore: local JSON writer, the source of truth (CLAUDE.md §1.3). The
only TraceStore implementation the orchestrator is required to succeed
against — its failures are allowed to propagate, unlike Mongo/Redis.

One file per run_id, rewritten (not appended) on every save_round call so the
file on disk always reflects the trace's current state, even if the process
crashes mid-debate.
"""

from __future__ import annotations

import json
from pathlib import Path

from committee.models.trace import DebateTrace, RoundRecord


class JsonStore:
    def __init__(self, trace_json_dir: str):
        self.trace_json_dir = Path(trace_json_dir)
        self.trace_json_dir.mkdir(parents=True, exist_ok=True)
        self._in_progress: dict[str, DebateTrace] = {}

    def _path_for(self, run_id: str) -> Path:
        return self.trace_json_dir / f"{run_id}.json"

    async def save_round(self, run_id: str, round_record: RoundRecord) -> None:
        """Requires the trace to have been registered via `register_trace`
        (called by the orchestrator once at debate start) so round-by-round
        writes have something to attach to and rewrite."""
        trace = self._in_progress.get(run_id)
        if trace is None:
            raise ValueError(
                f"save_round called for unregistered run_id={run_id!r}; "
                "register_trace must be called once at debate start."
            )
        # Trace object is mutated in place by the orchestrator already (it
        # appends to trace.rounds itself) — this just persists current state.
        self._write(trace)

    async def save_final(self, run_id: str, trace: DebateTrace) -> None:
        self._in_progress[run_id] = trace
        self._write(trace)
        self._in_progress.pop(run_id, None)

    async def get_run(self, run_id: str) -> DebateTrace | None:
        path = self._path_for(run_id)
        if not path.exists():
            return None
        return DebateTrace.model_validate_json(path.read_text())

    async def list_runs(self) -> list[str]:
        return sorted(p.stem for p in self.trace_json_dir.glob("*.json"))

    def register_trace(self, run_id: str, trace: DebateTrace) -> None:
        """Called once by the orchestrator at debate start, before any round
        completes, so save_round has a trace object to persist each round."""
        self._in_progress[run_id] = trace

    def _write(self, trace: DebateTrace) -> None:
        path = self._path_for(trace.run_id)
        path.write_text(trace.model_dump_json(indent=2))
