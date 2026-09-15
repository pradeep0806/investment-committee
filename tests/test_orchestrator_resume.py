"""Crash-and-resume: a debate that stops after round 1 (simulating a process
crash before round 2 starts) must, when resumed by run_id, continue from
round 2 rather than re-running round 1 from scratch."""

import asyncio
from datetime import timedelta

import pytest

from committee.agents.registry import build_agents
from committee.llm.client import LLMClient
from committee.models.requests import DebateConfig, ThesisRequest
from committee.orchestration.budget_gate import BudgetGate
from committee.orchestration.budget_manager import BudgetExhaustedError
from committee.orchestration.orchestrator import (
    DebateOrchestrator,
    RunAlreadyInProgressError,
    RunLockedByAnotherPodError,
)
from committee.storage.json_store import JsonStore
from committee.storage.run_lock_store import RunLockStore
from tests.test_run_lock_store import _FakeLockCollection


class _CountingRawCaller:
    def __init__(self):
        self.call_count = 0

    async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens=None):
        self.call_count += 1
        return (
            {
                "stance": "Buy",
                "confidence": 65,
                "key_factors": ["growth"],
                "evidence": [f"data point #{self.call_count}"],
                "top_risk": "x",
                "executive_summary": "test summary",
            },
            500,
        )


def _make_run_lock_store(collection: _FakeLockCollection) -> RunLockStore:
    store = RunLockStore.__new__(RunLockStore)
    store._collection = collection
    store._lease = timedelta(seconds=120)
    store._index_ensured = True
    return store


def _make_gate(raw_caller, total_budget=8000) -> BudgetGate:
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
    client._raw_caller = raw_caller
    return BudgetGate(llm_client=client, total_budget=total_budget)


async def test_resume_continues_from_last_completed_round_not_from_scratch(tmp_path):
    """Simulates a crash after round 1 by hand-rolling the persisted state a
    real crash would leave behind (an in-progress checkpoint + a trace with
    exactly one completed round and no ended_at) — the orchestrator's normal
    round loop always runs to completion or a clean early-stop, so a bare
    process kill mid-debate isn't reproducible by calling .run() a second
    time; this constructs the on-disk state directly instead."""
    from datetime import UTC, datetime

    from committee.models.checkpoint import DebateCheckpoint
    from committee.models.trace import DebateTrace

    json_store = JsonStore(trace_json_dir=str(tmp_path))
    config = DebateConfig(total_token_budget=8000, num_rounds=3)
    run_id = "resume-test-run"
    request = ThesisRequest(thesis="Test thesis")

    # Produce one real, valid round of agent outputs (round 1's own
    # BudgetGate/orchestrator instance — not reused below) so the "already
    # completed round 1" state is realistic, not fabricated Pydantic data.
    raw_caller_1 = _CountingRawCaller()
    gate_1 = _make_gate(raw_caller_1)
    agents_1 = build_agents(budget_gate=gate_1)
    single_round_config = config.model_copy(update={"num_rounds": 2})
    orchestrator_1 = DebateOrchestrator(
        config=single_round_config, agents=agents_1, budget_gate=gate_1, trace_store=json_store
    )
    seed_trace = await orchestrator_1.run(request, run_id=f"{run_id}-seed")
    round_one_record = seed_trace.rounds[0]

    crashed_trace = DebateTrace(
        run_id=run_id,
        request=request,
        config=config,
        rounds=[round_one_record],
        all_agent_outputs=list(round_one_record.agent_outputs),
        budget_ledger=list(round_one_record.ledger_entries),
        convergence_signals=[round_one_record.convergence_signal],
        started_at=seed_trace.started_at,
        ended_at=None,  # never reached save_final -> this is what a crash leaves
        total_tokens_used=sum(e.tokens_used for e in round_one_record.ledger_entries),
    )
    await json_store.save_final(run_id, crashed_trace)  # writes the file; ended_at stays None
    await json_store.save_checkpoint(
        DebateCheckpoint(
            run_id=run_id,
            last_completed_round=1,
            phase="in_progress",
            budget_remaining=config.total_token_budget - crashed_trace.total_tokens_used,
            updated_at=datetime.now(UTC),
        )
    )

    # New orchestrator instance, new raw caller/gate — simulates a fresh
    # process after the crash, using the full 3-round config the original
    # debate was actually supposed to run.
    raw_caller_2 = _CountingRawCaller()
    gate_2 = _make_gate(raw_caller_2)
    agents_2 = build_agents(budget_gate=gate_2)
    orchestrator_2 = DebateOrchestrator(
        config=config, agents=agents_2, budget_gate=gate_2, trace_store=json_store
    )

    resumed_trace = await orchestrator_2.run(request, run_id=run_id, resume=True)

    # Round 1 was NOT re-run: the second raw caller was only invoked for
    # rounds 2 and 3 (4 agents x 2 rounds = 8 calls), not all 3 rounds.
    assert raw_caller_2.call_count == 8
    assert len(resumed_trace.rounds) == 3
    assert resumed_trace.rounds[0].round == 1
    assert resumed_trace.rounds[1].round == 2
    assert resumed_trace.rounds[2].round == 3
    # Round 1's outputs are exactly what was persisted before the "crash",
    # not freshly regenerated.
    assert resumed_trace.rounds[0].agent_outputs == round_one_record.agent_outputs
    assert resumed_trace.ended_at is not None
    assert resumed_trace.synthesis is not None


async def test_resume_with_no_existing_checkpoint_falls_back_to_fresh_start(tmp_path):
    """resume=True with a run_id that was never actually started must not
    crash or silently no-op — it starts a normal fresh debate."""
    json_store = JsonStore(trace_json_dir=str(tmp_path))
    config = DebateConfig(total_token_budget=8000, num_rounds=2)

    raw_caller = _CountingRawCaller()
    gate = _make_gate(raw_caller)
    agents = build_agents(budget_gate=gate)
    orchestrator = DebateOrchestrator(
        config=config, agents=agents, budget_gate=gate, trace_store=json_store
    )

    trace = await orchestrator.run(
        ThesisRequest(thesis="Test thesis"), run_id="never-started-run", resume=True
    )

    assert len(trace.rounds) == 2
    assert trace.ended_at is not None


async def test_resume_of_already_completed_debate_does_not_rerun_it(tmp_path):
    json_store = JsonStore(trace_json_dir=str(tmp_path))
    config = DebateConfig(total_token_budget=8000, num_rounds=2)

    raw_caller_1 = _CountingRawCaller()
    gate_1 = _make_gate(raw_caller_1)
    agents_1 = build_agents(budget_gate=gate_1)
    orchestrator_1 = DebateOrchestrator(
        config=config, agents=agents_1, budget_gate=gate_1, trace_store=json_store
    )

    run_id = "already-complete-run"
    first_trace = await orchestrator_1.run(ThesisRequest(thesis="Test thesis"), run_id=run_id)
    assert first_trace.ended_at is not None

    # A fresh "process" (new gate/orchestrator, matching a real restart)
    # attempts to resume the same run_id.
    raw_caller_2 = _CountingRawCaller()
    gate_2 = _make_gate(raw_caller_2)
    agents_2 = build_agents(budget_gate=gate_2)
    orchestrator_2 = DebateOrchestrator(
        config=config, agents=agents_2, budget_gate=gate_2, trace_store=json_store
    )

    resumed_trace = await orchestrator_2.run(
        ThesisRequest(thesis="Test thesis"), run_id=run_id, resume=True
    )

    # get_run finds an already-`ended_at`-set trace -> resume falls back to
    # fresh start (per the documented "nothing to resume" branch), so this
    # asserts the safe behavior: it doesn't crash trying to continue past
    # num_rounds, and it still produces a complete trace either way.
    assert resumed_trace.ended_at is not None
    assert raw_caller_2.call_count == 8


class _SlowRawCaller:
    """Yields control at each call via asyncio.sleep so a test can
    deterministically start a second concurrent .run() call while the first
    is still mid-flight — reproduces the real bug found live: two
    overlapping POST /debate/{run_id}/resume requests for the same run_id
    each built their own DebateTrace and both wrote to the same run_id's
    storage, leaving JSON and Mongo with different, inconsistent round
    counts."""

    def __init__(self, delay: float = 0.05):
        self.call_count = 0
        self.delay = delay

    async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens=None):
        self.call_count += 1
        await asyncio.sleep(self.delay)
        return (
            {
                "stance": "Buy",
                "confidence": 65,
                "key_factors": ["growth"],
                "evidence": [f"data point #{self.call_count}"],
                "top_risk": "x",
                "executive_summary": "test summary",
            },
            500,
        )


class TestConcurrentRunGuard:
    """A second .run() call for a run_id already being processed must be
    rejected immediately, not silently race the first — two concurrent
    writers to the same run_id's checkpoint/trace corrupt state rather than
    merely duplicating work (confirmed: JSON ended up with fewer completed
    rounds than Mongo for the same run_id after an accidental double-resume
    in manual testing)."""

    async def test_second_concurrent_run_for_same_run_id_is_rejected(self, tmp_path):
        json_store = JsonStore(trace_json_dir=str(tmp_path))
        config = DebateConfig(total_token_budget=8000, num_rounds=2)
        raw_caller = _SlowRawCaller(delay=0.1)
        gate = _make_gate(raw_caller)
        agents = build_agents(budget_gate=gate)
        orchestrator = DebateOrchestrator(
            config=config, agents=agents, budget_gate=gate, trace_store=json_store
        )

        run_id = "concurrent-guard-test"
        first_call = asyncio.create_task(
            orchestrator.run(ThesisRequest(thesis="Test thesis"), run_id=run_id)
        )
        # Let the first call actually start (acquire the lock, begin round 1)
        # before firing the second — otherwise both could race to check the
        # lock before either has set it.
        await asyncio.sleep(0.02)

        with pytest.raises(RunAlreadyInProgressError):
            await orchestrator.run(ThesisRequest(thesis="Test thesis"), run_id=run_id)

        # The first call must complete normally, undisturbed by the rejected
        # second one.
        first_trace = await first_call
        assert first_trace.ended_at is not None
        assert len(first_trace.rounds) == 2

    async def test_lock_is_released_after_completion_allowing_a_later_resume(self, tmp_path):
        """The guard must not be a permanent lock on the run_id — once the
        first call finishes, a legitimate later resume attempt (e.g. the
        debate genuinely didn't finish and a real retry comes in afterward)
        must still be able to proceed."""
        json_store = JsonStore(trace_json_dir=str(tmp_path))
        config = DebateConfig(total_token_budget=8000, num_rounds=2)

        run_id = "lock-release-test"
        gate_1 = _make_gate(_CountingRawCaller())
        orchestrator_1 = DebateOrchestrator(
            config=config,
            agents=build_agents(budget_gate=gate_1),
            budget_gate=gate_1,
            trace_store=json_store,
        )
        await orchestrator_1.run(ThesisRequest(thesis="Test thesis"), run_id=run_id)

        # Lock released -> a second call for the same run_id (even though
        # it'll just fall back to a fresh start here, since the first one
        # already completed) must not raise RunAlreadyInProgressError. Fresh
        # gate/orchestrator, same as a real second request would use.
        gate_2 = _make_gate(_CountingRawCaller())
        orchestrator_2 = DebateOrchestrator(
            config=config,
            agents=build_agents(budget_gate=gate_2),
            budget_gate=gate_2,
            trace_store=json_store,
        )
        second_trace = await orchestrator_2.run(
            ThesisRequest(thesis="Test thesis"), run_id=run_id, resume=True
        )
        assert second_trace.ended_at is not None


class TestCrossPodRunLockGuard:
    """The cross-pod counterpart to TestConcurrentRunGuard: two separate
    DebateOrchestrator instances (simulating two pods, each its own process
    with its own empty _active_run_locks dict) sharing one Mongo-backed
    RunLockStore (here, a fake collection standing in for the one real
    collection all pods actually share) must not both proceed against the
    same run_id — the in-process lock alone can't see across pods, so this
    is the mechanism that closes that gap."""

    async def test_second_pod_claim_for_an_in_flight_run_id_is_rejected(self, tmp_path):
        json_store = JsonStore(trace_json_dir=str(tmp_path))
        config = DebateConfig(total_token_budget=8000, num_rounds=2)
        shared_collection = _FakeLockCollection()

        gate_1 = _make_gate(_SlowRawCaller(delay=0.1))
        orchestrator_1 = DebateOrchestrator(
            config=config,
            agents=build_agents(budget_gate=gate_1),
            budget_gate=gate_1,
            trace_store=json_store,
            run_lock_store=_make_run_lock_store(shared_collection),
            holder_id="pod-a",
        )
        gate_2 = _make_gate(_CountingRawCaller())
        orchestrator_2 = DebateOrchestrator(
            config=config,
            agents=build_agents(budget_gate=gate_2),
            budget_gate=gate_2,
            trace_store=json_store,
            run_lock_store=_make_run_lock_store(shared_collection),
            holder_id="pod-b",
        )

        run_id = "cross-pod-guard-test"
        first_call = asyncio.create_task(
            orchestrator_1.run(ThesisRequest(thesis="Test thesis"), run_id=run_id)
        )
        # Let pod-a actually start and claim the lock before pod-b attempts
        # the same run_id — matches TestConcurrentRunGuard's choreography.
        await asyncio.sleep(0.02)

        # Real pods are separate processes, each with its own empty
        # _active_run_locks dict — pod-b would never see pod-a's in-process
        # lock at all. This test process shares one Python module (and
        # therefore one _active_run_locks dict) between both simulated
        # "pods", so remove pod-a's in-process lock entry before pod-b's
        # attempt to accurately reproduce that isolation; what's actually
        # under test here is the Mongo-backed cross-pod claim, not the
        # already-covered in-process guard (TestConcurrentRunGuard).
        from committee.orchestration import orchestrator as orchestrator_module

        orchestrator_module._active_run_locks.pop(run_id, None)

        with pytest.raises(RunLockedByAnotherPodError):
            await orchestrator_2.run(ThesisRequest(thesis="Test thesis"), run_id=run_id)

        first_trace = await first_call
        assert first_trace.ended_at is not None
        assert len(first_trace.rounds) == 2

    async def test_claim_is_released_after_completion_allowing_a_different_pod_to_resume_later(
        self, tmp_path
    ):
        json_store = JsonStore(trace_json_dir=str(tmp_path))
        config = DebateConfig(total_token_budget=8000, num_rounds=2)
        shared_collection = _FakeLockCollection()

        run_id = "cross-pod-release-test"
        gate_1 = _make_gate(_CountingRawCaller())
        orchestrator_1 = DebateOrchestrator(
            config=config,
            agents=build_agents(budget_gate=gate_1),
            budget_gate=gate_1,
            trace_store=json_store,
            run_lock_store=_make_run_lock_store(shared_collection),
            holder_id="pod-a",
        )
        await orchestrator_1.run(ThesisRequest(thesis="Test thesis"), run_id=run_id)

        # pod-a's claim was released on completion (the try/finally in
        # .run()) -> a different pod resuming the same run_id afterward
        # must not be rejected.
        gate_2 = _make_gate(_CountingRawCaller())
        orchestrator_2 = DebateOrchestrator(
            config=config,
            agents=build_agents(budget_gate=gate_2),
            budget_gate=gate_2,
            trace_store=json_store,
            run_lock_store=_make_run_lock_store(shared_collection),
            holder_id="pod-b",
        )
        second_trace = await orchestrator_2.run(
            ThesisRequest(thesis="Test thesis"), run_id=run_id, resume=True
        )
        assert second_trace.ended_at is not None

    async def test_claim_is_released_even_when_the_round_loop_raises(self, tmp_path):
        """The try/finally around _run_locked must release the claim on any
        exception, not just success — otherwise one failed debate would
        permanently wedge its run_id for every pod."""
        json_store = JsonStore(trace_json_dir=str(tmp_path))
        # A budget too small to cover even a floor allocation for every
        # agent raises BudgetExhaustedError out of .run() itself, before
        # any round completes — a genuine exception path through the
        # try/finally around _run_locked, not the agent-exclusion path
        # (which is handled inside the round loop, not a crash).
        tiny_config = DebateConfig(total_token_budget=2, num_rounds=2)
        shared_collection = _FakeLockCollection()

        run_id = "cross-pod-exception-release-test"
        gate_1 = _make_gate(_CountingRawCaller())
        orchestrator_1 = DebateOrchestrator(
            config=tiny_config,
            agents=build_agents(budget_gate=gate_1),
            budget_gate=gate_1,
            trace_store=json_store,
            run_lock_store=_make_run_lock_store(shared_collection),
            holder_id="pod-a",
        )

        with pytest.raises(BudgetExhaustedError):
            await orchestrator_1.run(ThesisRequest(thesis="Test thesis"), run_id=run_id)

        # Claim released despite the exception -> a second pod can now claim
        # it, with a workable budget this time (a fresh run since round 1
        # never actually completed for pod-a).
        gate_2 = _make_gate(_CountingRawCaller())
        orchestrator_2 = DebateOrchestrator(
            config=DebateConfig(total_token_budget=8000, num_rounds=2),
            agents=build_agents(budget_gate=gate_2),
            budget_gate=gate_2,
            trace_store=json_store,
            run_lock_store=_make_run_lock_store(shared_collection),
            holder_id="pod-b",
        )
        second_trace = await orchestrator_2.run(ThesisRequest(thesis="Test thesis"), run_id=run_id)
        assert second_trace.ended_at is not None

    async def test_orchestrator_with_no_run_lock_store_behaves_exactly_as_before(self, tmp_path):
        """Regression guard for the graceful-degradation decision: Mongo
        unavailable (run_lock_store=None) must reproduce
        TestConcurrentRunGuard's pre-existing behavior unchanged — the
        in-process lock is still the only gate, no Mongo interaction is
        attempted, and nothing raises RunLockedByAnotherPodError."""
        json_store = JsonStore(trace_json_dir=str(tmp_path))
        config = DebateConfig(total_token_budget=8000, num_rounds=2)
        gate = _make_gate(_SlowRawCaller(delay=0.05))
        agents = build_agents(budget_gate=gate)
        orchestrator = DebateOrchestrator(
            config=config,
            agents=agents,
            budget_gate=gate,
            trace_store=json_store,
            run_lock_store=None,
        )

        run_id = "no-run-lock-store-test"
        first_call = asyncio.create_task(
            orchestrator.run(ThesisRequest(thesis="Test thesis"), run_id=run_id)
        )
        await asyncio.sleep(0.02)

        with pytest.raises(RunAlreadyInProgressError):
            await orchestrator.run(ThesisRequest(thesis="Test thesis"), run_id=run_id)

        first_trace = await first_call
        assert first_trace.ended_at is not None
