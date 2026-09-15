import json
from datetime import datetime, timezone

import pytest

from committee.models.agent_output import AgentOutput, Stance
from committee.models.requests import DebateConfig, ThesisRequest
from committee.models.synthesis import ConvergenceSignal
from committee.models.trace import BudgetLedgerEntry, DebateTrace, RoundRecord
from committee.storage.best_effort import BestEffortTraceStore
from committee.storage.json_store import JsonStore


def _sample_trace(run_id: str = "run-1") -> DebateTrace:
    now = datetime.now(timezone.utc)
    output = AgentOutput(
        agent_id="fundamentals",
        round=1,
        stance=Stance.BUY,
        confidence=80,
        key_factors=["growth"],
        top_risk="risk",
        tokens_used=500,
    )
    signal = ConvergenceSignal(
        round=1,
        stance_agreement=1.0,
        factor_overlap=1.0,
        confidence_spread=0.0,
        composite_score=1.0,
        mode_selected="exploit",
    )
    ledger_entry = BudgetLedgerEntry(
        round=1, agent_id="fundamentals", tokens_allocated=1000, tokens_used=500, mode="exploit"
    )
    round_record = RoundRecord(
        round=1,
        agent_outputs=[output],
        convergence_signal=signal,
        ledger_entries=[ledger_entry],
        started_at=now,
        ended_at=now,
    )
    return DebateTrace(
        run_id=run_id,
        request=ThesisRequest(thesis="Test thesis"),
        config=DebateConfig(),
        rounds=[round_record],
        all_agent_outputs=[output],
        budget_ledger=[ledger_entry],
        convergence_signals=[signal],
        started_at=now,
    )


class TestJsonStore:
    async def test_save_final_writes_a_readable_file(self, tmp_path):
        store = JsonStore(trace_json_dir=str(tmp_path))
        trace = _sample_trace()

        await store.save_final(trace.run_id, trace)

        path = tmp_path / f"{trace.run_id}.json"
        assert path.exists()
        on_disk = json.loads(path.read_text())
        assert on_disk["run_id"] == trace.run_id

    async def test_get_run_round_trips_the_trace(self, tmp_path):
        store = JsonStore(trace_json_dir=str(tmp_path))
        trace = _sample_trace()
        await store.save_final(trace.run_id, trace)

        restored = await store.get_run(trace.run_id)

        assert restored == trace

    async def test_get_run_returns_none_for_unknown_run_id(self, tmp_path):
        store = JsonStore(trace_json_dir=str(tmp_path))
        assert await store.get_run("does-not-exist") is None

    async def test_list_runs_returns_all_saved_run_ids(self, tmp_path):
        store = JsonStore(trace_json_dir=str(tmp_path))
        await store.save_final("run-a", _sample_trace("run-a"))
        await store.save_final("run-b", _sample_trace("run-b"))

        run_ids = await store.list_runs()

        assert run_ids == ["run-a", "run-b"]

    async def test_save_round_rewrites_the_file_with_current_state(self, tmp_path):
        store = JsonStore(trace_json_dir=str(tmp_path))
        trace = _sample_trace()
        store.register_trace(trace.run_id, trace)

        await store.save_round(trace.run_id, trace.rounds[0])

        restored = await store.get_run(trace.run_id)
        assert restored is not None
        assert len(restored.rounds) == 1

    async def test_save_round_without_registration_raises(self, tmp_path):
        store = JsonStore(trace_json_dir=str(tmp_path))
        trace = _sample_trace()
        with pytest.raises(ValueError):
            await store.save_round(trace.run_id, trace.rounds[0])


class _AlwaysFailingBackend:
    """Simulates an unreachable Mongo/Redis — every method raises."""

    async def save_round(self, run_id, round_record):
        raise ConnectionError("mongo unreachable")

    async def save_final(self, run_id, trace):
        raise ConnectionError("mongo unreachable")

    async def get_run(self, run_id):
        raise ConnectionError("mongo unreachable")

    async def list_runs(self):
        raise ConnectionError("mongo unreachable")


class TestBestEffortTraceStore:
    async def test_save_round_swallows_backend_failure(self):
        wrapped = BestEffortTraceStore(_AlwaysFailingBackend(), backend_name="mongo")
        trace = _sample_trace()
        # Must not raise.
        await wrapped.save_round(trace.run_id, trace.rounds[0])

    async def test_save_final_swallows_backend_failure(self):
        wrapped = BestEffortTraceStore(_AlwaysFailingBackend(), backend_name="mongo")
        trace = _sample_trace()
        # Must not raise.
        await wrapped.save_final(trace.run_id, trace)

    async def test_get_run_failure_propagates(self):
        """Unlike writes, a read failure is not silently swallowed — a caller
        explicitly asking to read a run deserves to know it failed."""
        wrapped = BestEffortTraceStore(_AlwaysFailingBackend(), backend_name="mongo")
        with pytest.raises(ConnectionError):
            await wrapped.get_run("some-run")


class TestOrchestratorStorageIntegration:
    async def test_debate_completes_and_writes_json_even_when_mongo_and_redis_always_fail(self, tmp_path):
        """The hard requirement (CLAUDE.md §1.3, §5 step 7): Mongo/Redis
        failures never block the debate. JSON is always written regardless."""
        from committee.agents.registry import build_agents
        from committee.orchestration.budget_gate import BudgetGate
        from committee.orchestration.orchestrator import DebateOrchestrator

        class _FakeRawCaller:
            async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens=None):
                return (
                    {
                        "stance": "Buy",
                        "confidence": 65,
                        "key_factors": ["growth"],
                        "evidence": ["Q3 revenue up 22% YoY"],
                        "top_risk": "x",
                    },
                    500,
                )

        from committee.llm.client import LLMClient

        client = LLMClient.__new__(LLMClient)
        client.provider = "fake"
        client.model = "fake-model"
        client.api_key = "fake-key"
        client.timeout_seconds = 60
        client.max_retries = 3
        client.retry_backoff_seconds = 0
        client.fallback_provider = None
        client.fallback_model = None
        client.fallback_max_retries = 0
        client._fallback_raw_caller = None
        client._raw_caller = _FakeRawCaller()

        config = DebateConfig(total_token_budget=8000, num_rounds=2)
        gate = BudgetGate(llm_client=client, total_budget=config.total_token_budget)
        agents = build_agents(budget_gate=gate)

        json_store = JsonStore(trace_json_dir=str(tmp_path))
        failing_mongo = BestEffortTraceStore(_AlwaysFailingBackend(), backend_name="mongo")

        class _DualStore:
            def __init__(self, json_store, mongo_store):
                self._json = json_store
                self._mongo = mongo_store

            async def save_round(self, run_id, round_record):
                await self._json.save_round(run_id, round_record)
                await self._mongo.save_round(run_id, round_record)

            async def save_final(self, run_id, trace):
                await self._json.save_final(run_id, trace)
                await self._mongo.save_final(run_id, trace)

            async def get_run(self, run_id):
                return await self._json.get_run(run_id)

            async def list_runs(self):
                return await self._json.list_runs()

            def register_trace(self, run_id, trace):
                self._json.register_trace(run_id, trace)

        dual_store = _DualStore(json_store, failing_mongo)
        orchestrator = DebateOrchestrator(
            config=config,
            agents=agents,
            trace_store=dual_store,
            redis_bus=None,  # orchestrator only calls redis_bus if not None; simulate absence
        )

        trace = await orchestrator.run(ThesisRequest(thesis="Test thesis"))

        assert trace.ended_at is not None
        assert len(trace.rounds) == 2

        on_disk = await json_store.get_run(trace.run_id)
        assert on_disk is not None
        assert len(on_disk.rounds) == 2

    async def test_debate_completes_even_when_redis_bus_itself_raises(self, tmp_path):
        """Defensive at the orchestrator boundary, not just trusting RedisBus's
        own internal try/except — any redis_bus-shaped object that raises
        must not block the debate."""
        from committee.agents.registry import build_agents
        from committee.llm.client import LLMClient
        from committee.orchestration.budget_gate import BudgetGate
        from committee.orchestration.orchestrator import DebateOrchestrator

        class _FakeRawCaller:
            async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens=None):
                return (
                    {
                        "stance": "Buy",
                        "confidence": 65,
                        "key_factors": ["growth"],
                        "evidence": ["Q3 revenue up 22% YoY"],
                        "top_risk": "x",
                    },
                    500,
                )

        client = LLMClient.__new__(LLMClient)
        client.provider = "fake"
        client.model = "fake-model"
        client.api_key = "fake-key"
        client.timeout_seconds = 60
        client.max_retries = 3
        client.retry_backoff_seconds = 0
        client.fallback_provider = None
        client.fallback_model = None
        client.fallback_max_retries = 0
        client._fallback_raw_caller = None
        client._raw_caller = _FakeRawCaller()

        config = DebateConfig(total_token_budget=8000, num_rounds=2)
        gate = BudgetGate(llm_client=client, total_budget=config.total_token_budget)
        agents = build_agents(budget_gate=gate)
        json_store = JsonStore(trace_json_dir=str(tmp_path))

        class _RawlyFailingRedisBus:
            """Deliberately does NOT swallow its own errors, unlike the real
            RedisBus — simulates a misbehaving or alternate implementation."""

            async def publish_round_event(self, run_id, event):
                raise ConnectionError("redis unreachable")

        orchestrator = DebateOrchestrator(
            config=config,
            agents=agents,
            trace_store=json_store,
            redis_bus=_RawlyFailingRedisBus(),
        )

        trace = await orchestrator.run(ThesisRequest(thesis="Test thesis"))

        assert trace.ended_at is not None
        assert len(trace.rounds) == 2
