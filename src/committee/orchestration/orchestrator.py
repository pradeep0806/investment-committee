"""DebateOrchestrator: the core loop, with no web-framework dependency (CLAUDE.md §1.1).

CLI and API both call into this same plain Python object. As of step 5, the
loop runs the configured `config.num_rounds` (not a hardcoded 2), and mode
each round is a *computed* signal from ExploreExploitController — never a
hardcoded round number (CLAUDE.md §1.6). Budget reallocation in exploit mode
targets the controller's contested agents; explore mode injects the
controller's rotating "argue against majority" directive.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone

from typing import Any, Awaitable, Callable

from committee.agents.base import AnalystAgent
from committee.agents.tie_breaker import TieBreakerAgent
from committee.llm.structured_output import LLMValidationError
from committee.models.agent_output import AgentOutput
from committee.models.checkpoint import DebateCheckpoint
from committee.models.requests import DebateConfig, ThesisRequest
from committee.models.trace import DebateTrace, RoundRecord
from committee.observability.logging import get_run_logger
from committee.observability.metrics import (
    debate_active_agent,
    debate_convergence_score,
    debate_duration_seconds,
    debate_tokens_used_total,
    disagreements_detected_total,
    mode_transitions_total,
)
from committee.orchestration.budget_gate import BudgetGate
from committee.orchestration.budget_manager import BudgetExhaustedError, BudgetManager
from committee.orchestration.conflict_resolution.registry import build_strategy
from committee.orchestration.disagreement import detect as detect_disagreements
from committee.orchestration.explore_exploit import ExploreExploitController
from committee.storage.run_lock_store import HOLDER_ID
from committee.synthesis.synthesizer import synthesize

EventSink = Callable[[dict], Awaitable[None]]

# Guards against two concurrent .run() calls for the same run_id racing each
# other's checkpoint/trace writes — real bug found live: two overlapping
# POST /debate/{run_id}/resume requests for the same run_id (a client retry
# after a slow/dropped first request, in this case) each built their own
# in-memory DebateTrace from the same starting checkpoint and both wrote to
# the same run_id's JSON file and Mongo document, leaving JSON with 2
# completed rounds and Mongo's debate_traces with 3 — genuinely inconsistent
# state, not just a wasted duplicate run. Module-level (not per-instance)
# because a fresh DebateOrchestrator is constructed per request — the lock
# has to be shared across instances to mean anything. Covers one process
# only — two separate pods (k8s/api.yaml ships `replicas: 2`, HPA to 6) each
# have their own empty dict, so the same corruption is reproducible across
# pods. RunLockStore (storage/run_lock_store.py), consulted inside this same
# lock further down, is the cross-pod counterpart — same two-tier shape as
# BudgetGate's in-process lock + BudgetStore's atomic Mongo backing.
_active_run_locks: dict[str, asyncio.Lock] = {}


class RunAlreadyInProgressError(Exception):
    """Raised when .run() is called for a run_id that's already being
    processed by another concurrent call in this process."""


class RunLockedByAnotherPodError(Exception):
    """Raised when .run() is called for a run_id whose cross-pod Mongo
    claim (storage/run_lock_store.py) is currently held by a different
    process/pod — the multi-replica counterpart to RunAlreadyInProgressError,
    kept as a distinct type since the two have different operational
    meanings (this pod vs. some other pod) even though both map to the same
    HTTP 409 at the API layer."""


class DebateOrchestrator:
    def __init__(
        self,
        config: DebateConfig,
        agents: list[AnalystAgent],
        controller: ExploreExploitController | None = None,
        budget_gate: BudgetGate | None = None,
        budget_store: Any | None = None,
        run_lock_store: Any | None = None,
        trace_store: Any | None = None,
        redis_bus: Any | None = None,
        mlflow_tracker: Any | None = None,
        event_sink: EventSink | None = None,
        holder_id: str | None = None,
    ):
        self.config = config
        self.agents = agents
        # None means "use ExploreExploitController's own default" (0.4/0.75)
        # — this fallback path only actually runs for a caller that builds a
        # DebateOrchestrator directly (tests, mainly); orchestrator_factory.py
        # always constructs and passes an explicit controller, resolving
        # None against Settings.convergence_*_threshold before getting here.
        controller_kwargs: dict[str, float] = {}
        if config.convergence_low_threshold is not None:
            controller_kwargs["low_threshold"] = config.convergence_low_threshold
        if config.convergence_high_threshold is not None:
            controller_kwargs["high_threshold"] = config.convergence_high_threshold
        self.controller = controller or ExploreExploitController(**controller_kwargs)
        # Only needed if conflict_resolution_strategy=tie_breaker actually
        # ends up spawning an agent; every other strategy ignores it. Never
        # a raw LLMClient — the gate is the only reachable path to the LLM,
        # for the tie-breaker exactly as for every standing agent.
        self.budget_gate = budget_gate
        # Backs budget_gate's reservations with an atomic Mongo $inc once a
        # run_id is known (see BudgetGate.bind_run, called below in run()) —
        # optional, matching every other Mongo-touching piece here: absent
        # or unreachable, the gate still fully enforces budget in-process.
        self.budget_store = budget_store
        # Cross-pod counterpart to _active_run_locks — optional and
        # degrades the same way budget_store does: absent or unreachable,
        # the in-process lock still fully protects a single pod against
        # itself, just not against a different pod (see run_lock_store.py).
        self.run_lock_store = run_lock_store
        # Defaults to the true per-process identity; only ever overridden
        # in tests, to simulate two different "pods" sharing one fake Mongo
        # collection — orchestrator_factory.py never passes this.
        self._holder_id = holder_id or HOLDER_ID
        # All optional and independent of each other — JsonStore is the only
        # one anything actually depends on being present (source of truth);
        # trace_store here is expected to be JsonStore itself for the always-
        # blocking write, wrapped separately for any best-effort backend the
        # caller also wants (see storage/best_effort.py). redis_bus and
        # mlflow_tracker are pure side channels with no read dependency.
        self.trace_store = trace_store
        self.redis_bus = redis_bus
        self.mlflow_tracker = mlflow_tracker
        # The single hook both the SSE endpoint (step 9) and any other live
        # consumer subscribe to — fired at the same points RedisBus is
        # published to, so streaming a debate never means recomputing
        # anything the orchestration loop hasn't already computed once.
        self.event_sink = event_sink

    async def _emit(self, event: dict) -> None:
        if self.event_sink is None:
            return
        try:
            await self.event_sink(event)
        except Exception as exc:
            get_run_logger(event.get("run_id", "unknown")).warning(
                "event_sink_failed", error=str(exc)
            )

    async def run(
        self, request: ThesisRequest, run_id: str | None = None, resume: bool = False
    ) -> DebateTrace:
        """`run_id`/`resume`: the crash-recovery path. Passing an existing
        `run_id` with `resume=True` looks up that debate's last checkpoint
        and trace via `trace_store` and, if the debate is incomplete,
        continues from `checkpoint.last_completed_round + 1` instead of
        starting over — completed rounds' agent outputs are reloaded from
        the trace (the source of truth) rather than re-run, so a crash
        doesn't re-spend tokens on work that already finished. Omitting
        both (the default) is the normal fresh-start path, unchanged.

        Raises RunAlreadyInProgressError immediately (no queueing, no
        waiting) if another concurrent call for this same run_id is already
        in flight — two callers racing to resume (or, in principle, resume
        while a fresh run under an explicit run_id is still active) must
        never both proceed: each would build its own DebateTrace from the
        same starting point and both write to the same run_id's storage,
        which corrupts state rather than merely wasting work. Only applies
        when run_id is explicitly given; a fresh start with run_id=None has
        nothing to collide with until its uuid4() is generated below.
        """
        if run_id is not None:
            lock = _active_run_locks.setdefault(run_id, asyncio.Lock())
            if lock.locked():
                raise RunAlreadyInProgressError(
                    f"run_id={run_id!r} is already being processed by another call "
                    "in this process."
                )
            async with lock:
                if self.run_lock_store is not None:
                    acquired = await self.run_lock_store.acquire(run_id, self._holder_id)
                    if not acquired:
                        raise RunLockedByAnotherPodError(
                            f"run_id={run_id!r} is currently claimed by another "
                            "pod/process."
                        )
                    try:
                        return await self._run_locked(request, run_id, resume)
                    finally:
                        # Covers normal completion and any exception raised
                        # inside _run_locked. A hard process crash (no code
                        # runs at all) is NOT covered here — only the
                        # heartbeat/lease staleness check in acquire() can
                        # reclaim a run_id whose holder simply vanished.
                        await self.run_lock_store.release(run_id, self._holder_id)
                return await self._run_locked(request, run_id, resume)
        return await self._run_locked(request, run_id, resume)

    async def _run_locked(
        self, request: ThesisRequest, run_id: str | None, resume: bool
    ) -> DebateTrace:
        start_round = 1
        trace: DebateTrace | None = None

        if resume and run_id is not None and self.trace_store is not None:
            trace = await self.trace_store.get_run(run_id)
            checkpoint = None
            if hasattr(self.trace_store, "get_checkpoint"):
                checkpoint = await self.trace_store.get_checkpoint(run_id)
            if trace is not None and trace.ended_at is None and checkpoint is not None:
                start_round = checkpoint.last_completed_round + 1
            else:
                # Nothing to resume (already finished, or no checkpoint ever
                # written) — fall back to a normal fresh start under the
                # same run_id rather than silently no-opping.
                trace = None

        if trace is None:
            trace = DebateTrace(
                run_id=run_id or str(uuid.uuid4()),
                request=request,
                config=self.config,
                started_at=datetime.now(timezone.utc),
            )
        # trace.run_id (unlike the run_id parameter) is always a str from
        # here on — resolved above whether this is a fresh start or resume.
        run_id = trace.run_id

        logger = get_run_logger(run_id)

        if self.budget_gate is not None:
            self.budget_gate.bind_run(run_id, self.budget_store)

        # None means "use BudgetManager's own default" (256, matching
        # structured_output.py's MIN_MAX_TOKENS) — this fallback path only
        # actually runs for a caller that builds a DebateOrchestrator
        # directly (tests, mainly); orchestrator_factory.py always resolves
        # None against Settings.min_viable_allocation_per_agent before a
        # config gets here, same pattern as convergence_low_threshold above.
        budget_manager_kwargs: dict[str, int] = {}
        if self.config.min_viable_allocation_per_agent is not None:
            budget_manager_kwargs["min_viable_allocation"] = self.config.min_viable_allocation_per_agent
        budget_manager = BudgetManager(
            total_token_budget=self.config.total_token_budget,
            num_rounds=self.config.num_rounds,
            num_agents=len(self.agents),
            **budget_manager_kwargs,
        )
        # budget_gate (when provided) is constructed by the same caller, once
        # per debate, with the same total_token_budget (see
        # orchestrator_factory.py) — the two start in sync. BudgetManager
        # decides *how much* each agent gets allocated per round (planning);
        # BudgetGate is the structural backstop that makes it impossible for
        # any call to actually spend more than what remains, independent of
        # whether BudgetManager's allocation math is ever wrong or bypassed.
        # On resume, budget already spent in completed rounds must not be
        # re-granted — replay it into both trackers before round start_round
        # runs, so remaining budget reflects everything actually spent
        # before the crash, not just what a fresh BudgetManager would assume.
        for round_record in trace.rounds:
            for entry in round_record.ledger_entries:
                budget_manager.record_actual_usage(
                    round=entry.round,
                    agent_id=entry.agent_id,
                    tokens_allocated=entry.tokens_allocated,
                    tokens_used=entry.tokens_used,
                    mode=entry.mode,
                    provider_used=entry.provider_used,
                    excluded=entry.excluded,
                )
        if self.budget_gate is not None and trace.rounds:
            already_spent = sum(entry.tokens_used for r in trace.rounds for entry in r.ledger_entries)
            if already_spent:
                await self.budget_gate._reserve(min(already_spent, self.budget_gate.remaining))

        if self.trace_store is not None and hasattr(self.trace_store, "register_trace"):
            self.trace_store.register_trace(run_id, trace)
        if self.mlflow_tracker is not None:
            # Defensive even though MlflowRunTracker already swallows its own
            # errors — same reasoning as redis_bus below: the "never blocks
            # the debate" contract is the orchestrator's to guarantee.
            try:
                self.mlflow_tracker.start_run(
                    request=request,
                    config=self.config,
                    model=self.budget_gate.model if self.budget_gate else "unknown",
                    num_agents=len(self.agents),
                )
            except Exception as exc:
                logger.warning("mlflow_tracker_start_run_failed", error=str(exc))

        start_time = time.monotonic()
        logger.info("debate_started", thesis=request.thesis, num_agents=len(self.agents))

        # On a fresh start these stay at their defaults (nothing prior
        # exists yet). On resume, they're seeded from the reloaded trace's
        # already-completed rounds so round `start_round` sees the same
        # prior-round context it would have if the process had never
        # crashed — completed rounds are not re-run, only reused.
        prior_round_outputs: list[AgentOutput] | None = None
        all_prior_outputs: list[AgentOutput] = []
        directives: dict[str, str] = {}
        previous_mode: str | None = None
        for round_record in trace.rounds:
            if prior_round_outputs is not None:
                all_prior_outputs = all_prior_outputs + prior_round_outputs
            prior_round_outputs = round_record.agent_outputs
            previous_mode = round_record.convergence_signal.mode_selected

        for round_num in range(start_round, self.config.num_rounds + 1):
            round_started_at = datetime.now(timezone.utc)

            # Round 1 has no prior outputs to score convergence on, so it
            # always starts in explore mode — there's nothing yet to converge
            # around. From round 2 onward, mode is decided from the *previous*
            # round's computed convergence signal. all_prior_outputs (every
            # round strictly before round_num - 1, i.e. everything except the
            # round just scored) is passed so the classifier can catch an
            # echo of something said two-plus rounds back, not just of the
            # immediately preceding round.
            if prior_round_outputs is None:
                mode = "explore"
                contested = []
            else:
                signal = self.controller.score(
                    prior_round_outputs,
                    round=round_num - 1,
                    all_prior_outputs=all_prior_outputs,
                )
                mode = self.controller.decide_mode(signal)
                contested = self.controller.contested_agents(prior_round_outputs)
                # prior_round_outputs is now folded into all_prior_outputs so
                # this round's *real* convergence scoring call (below, once
                # this round's own agent_outputs exist) has visibility into
                # every round strictly before round_num, not just round_num - 1.
                all_prior_outputs = all_prior_outputs + prior_round_outputs

            if previous_mode is not None and mode != previous_mode:
                mode_transitions_total.labels(from_mode=previous_mode, to_mode=mode).inc()
                await self._emit(
                    {
                        "event": "mode_transition",
                        "run_id": run_id,
                        "round": round_num,
                        "from_mode": previous_mode,
                        "to_mode": mode,
                    }
                )
            previous_mode = mode

            logger.info("round_started", round=round_num, mode=mode)
            await self._emit({"event": "round_start", "run_id": run_id, "round": round_num, "mode": mode})

            try:
                allocations = budget_manager.allocate(
                    round=round_num,
                    agent_ids=[agent.agent_id for agent in self.agents],
                    mode=mode,
                    contested_agents=contested,
                )
            except BudgetExhaustedError as exc:
                # Real gap found via a live debate: allocate()'s viable-
                # allocation floor (MIN_VIABLE_ALLOCATION, see
                # budget_manager.py) can now raise on round 1 itself for a
                # small enough total_token_budget, not only on a later
                # round after prior overspend — a case the pre-round guard
                # was written for (this exact except block already existed
                # for that later-round case; it just couldn't previously
                # trigger this early). Stop cleanly with whatever rounds
                # already completed (possibly zero) rather than letting
                # this propagate as an unhandled exception and crash the
                # whole debate.
                logger.warning(
                    "budget_exhausted_before_round",
                    round=round_num,
                    remaining_budget=budget_manager.remaining_budget(),
                    error=str(exc),
                )
                await self._emit(
                    {
                        "event": "budget_exhausted",
                        "run_id": run_id,
                        "round": round_num,
                        "remaining_budget": budget_manager.remaining_budget(),
                    }
                )
                break

            agent_outputs: list[AgentOutput] = []
            for agent in self.agents:
                token_budget = allocations[agent.agent_id]
                directive = directives.get(agent.agent_id)
                logger.info("agent_reasoning_start", agent_id=agent.agent_id, round=round_num)
                debate_active_agent.labels(run_id=run_id, agent=agent.agent_id).set(1)
                await self._emit(
                    {
                        "event": "agent_reasoning_start",
                        "run_id": run_id,
                        "round": round_num,
                        "agent_id": agent.agent_id,
                    }
                )
                try:
                    output = await agent.analyze(
                        request=request,
                        round=round_num,
                        token_budget=token_budget,
                        prior_round_outputs=prior_round_outputs,
                        directive=directive,
                    )
                except LLMValidationError as exc:
                    logger.warning(
                        "agent_excluded_invalid_output",
                        agent_id=agent.agent_id,
                        round=round_num,
                        attempts=exc.attempts,
                        last_error=str(exc.last_error),
                        tokens_used=exc.total_tokens_used,
                    )
                    debate_active_agent.labels(run_id=run_id, agent=agent.agent_id).set(0)
                    await self._emit(
                        {
                            "event": "agent_reasoning_end",
                            "run_id": run_id,
                            "round": round_num,
                            "agent_id": agent.agent_id,
                            "excluded": True,
                        }
                    )
                    if exc.total_tokens_used:
                        # Real spend across the failed attempts is never
                        # discarded — BudgetGate.call() already reconciled
                        # its own reservation against exc.total_tokens_used
                        # (see budget_gate.py); this ledger entry is what
                        # makes that spend visible in the trace too, instead
                        # of an excluded agent silently leaving no record at
                        # all despite real tokens having been spent against
                        # the real provider.
                        budget_manager.record_actual_usage(
                            round=round_num,
                            agent_id=agent.agent_id,
                            tokens_allocated=token_budget,
                            tokens_used=exc.total_tokens_used,
                            mode=mode,
                            excluded=True,
                        )
                        debate_tokens_used_total.labels(
                            agent=agent.agent_id, round=str(round_num)
                        ).inc(exc.total_tokens_used)
                    continue
                except BudgetExhaustedError as exc:
                    # The gate refused this call outright (no network call
                    # made) — either allocate()'s own per-round math already
                    # accounted for what was left, or (a real bug found
                    # while fixing structured_output.py's retry-accumulation
                    # overrun in the same session) an *earlier* agent's real
                    # usage this round already overran its own reservation
                    # by enough to exhaust the gate mid-round, something
                    # BudgetGate previously never surfaced because it
                    # silently absorbed any overrun beyond what was
                    # reserved. Since no call was made, there's nothing to
                    # record in the ledger for this agent — but the debate
                    # must stop cleanly here rather than let this propagate
                    # as an unhandled exception and crash the whole run.
                    logger.warning(
                        "agent_excluded_budget_exhausted",
                        agent_id=agent.agent_id,
                        round=round_num,
                        error=str(exc),
                    )
                    debate_active_agent.labels(run_id=run_id, agent=agent.agent_id).set(0)
                    await self._emit(
                        {
                            "event": "agent_reasoning_end",
                            "run_id": run_id,
                            "round": round_num,
                            "agent_id": agent.agent_id,
                            "excluded": True,
                        }
                    )
                    await self._emit(
                        {
                            "event": "budget_exhausted",
                            "run_id": run_id,
                            "round": round_num,
                            "remaining_budget": budget_manager.remaining_budget(),
                        }
                    )
                    break

                debate_active_agent.labels(run_id=run_id, agent=agent.agent_id).set(0)
                logger.info(
                    "agent_reasoning_end",
                    agent_id=agent.agent_id,
                    round=round_num,
                    stance=output.stance.value,
                    tokens_used=output.tokens_used,
                )
                await self._emit(
                    {
                        "event": "agent_reasoning_end",
                        "run_id": run_id,
                        "round": round_num,
                        "agent_id": agent.agent_id,
                        "stance": output.stance.value,
                        "confidence": output.confidence,
                        "tokens_used": output.tokens_used,
                        "executive_summary": output.executive_summary,
                    }
                )
                agent_outputs.append(output)
                budget_manager.record_actual_usage(
                    round=round_num,
                    agent_id=agent.agent_id,
                    tokens_allocated=token_budget,
                    tokens_used=output.tokens_used,
                    mode=mode,
                    provider_used=output.provider_used,
                )
                debate_tokens_used_total.labels(agent=agent.agent_id, round=str(round_num)).inc(
                    output.tokens_used
                )

            disagreements = detect_disagreements(round=round_num, agent_outputs=agent_outputs)
            if disagreements:
                disagreements_detected_total.labels(run_id=run_id).inc(len(disagreements))
                logger.info(
                    "disagreement_detected",
                    round=round_num,
                    count=len(disagreements),
                    agents_involved=[a for d in disagreements for a in d.agents_involved],
                )
                await self._emit(
                    {
                        "event": "disagreement_detected",
                        "run_id": run_id,
                        "round": round_num,
                        "count": len(disagreements),
                        "agents_involved": [a for d in disagreements for a in d.agents_involved],
                    }
                )

            convergence_signal = self.controller.score(
                agent_outputs, round=round_num, all_prior_outputs=all_prior_outputs
            )
            debate_convergence_score.labels(run_id=run_id).set(convergence_signal.composite_score)
            await self._emit(
                {
                    "event": "convergence_computed",
                    "run_id": run_id,
                    "round": round_num,
                    "composite_score": convergence_signal.composite_score,
                    "mode_selected": convergence_signal.mode_selected,
                }
            )
            logger.info(
                "convergence_computed",
                round=round_num,
                composite_score=convergence_signal.composite_score,
                mode_selected=convergence_signal.mode_selected,
            )

            round_record = RoundRecord(
                round=round_num,
                agent_outputs=agent_outputs,
                convergence_signal=convergence_signal,
                disagreements=disagreements,
                ledger_entries=[e for e in budget_manager.ledger if e.round == round_num],
                started_at=round_started_at,
                ended_at=datetime.now(timezone.utc),
            )
            trace.rounds.append(round_record)
            trace.all_agent_outputs.extend(agent_outputs)
            trace.convergence_signals.append(convergence_signal)
            trace.disagreements.extend(disagreements)

            # Every round writes to: local JSON (always, blocking, source of
            # truth), Mongo (best-effort, via whatever wraps trace_store),
            # Redis pub/sub (best-effort, for the streaming endpoint), and
            # MLflow's per-round metric history — all from this same point in
            # the loop, right after the numbers they report on were computed,
            # not recomputed from the trace afterward (CLAUDE.md §5 step 7).
            if self.trace_store is not None:
                await self.trace_store.save_round(run_id, round_record)
                if hasattr(self.trace_store, "save_checkpoint"):
                    await self.trace_store.save_checkpoint(
                        DebateCheckpoint(
                            run_id=run_id,
                            last_completed_round=round_num,
                            phase="in_progress",
                            # budget_gate.remaining (when a gate is wired in)
                            # is the actual enforced ceiling — the same
                            # number BudgetStore's atomic Mongo ledger
                            # tracks — so this field and debate_budgets.
                            # remaining always agree. budget_manager.
                            # remaining_budget() is deliberately NOT used
                            # here: it reports only the non-reserve
                            # "spendable" slice (total_token_budget minus the
                            # 10% carved out for tie-breaker spawns), a
                            # smaller and differently-scoped number that
                            # looked like a bug when compared side-by-side
                            # with the gate's reading of the same debate.
                            budget_remaining=(
                                self.budget_gate.remaining
                                if self.budget_gate is not None
                                else budget_manager.remaining_budget()
                            ),
                            updated_at=datetime.now(timezone.utc),
                        )
                    )
            if self.run_lock_store is not None:
                # Piggybacks on this same per-round point rather than a
                # separate background task — a round against a real LLM
                # takes seconds-to-minutes, far tighter than the lease
                # window, and a genuinely crashed pod naturally stops
                # heartbeating because it stops making round progress at
                # all. Defensive: a heartbeat failure must never abort an
                # in-progress debate, it only affects how soon this run_id
                # becomes reclaimable if this pod later dies.
                try:
                    await self.run_lock_store.heartbeat(run_id, self._holder_id)
                except Exception as exc:
                    logger.warning("run_lock_heartbeat_failed", round=round_num, error=str(exc))
            if self.redis_bus is not None:
                # Defensive even though RedisBus itself already swallows its
                # own errors — a caller could pass any redis_bus-shaped
                # object, and the contract ("never blocks the debate") is the
                # orchestrator's to guarantee, not something to trust a
                # specific implementation to uphold.
                try:
                    await self.redis_bus.publish_round_event(
                        run_id,
                        {
                            "event": "round_complete",
                            "round": round_num,
                            "mode": mode,
                            "composite_score": convergence_signal.composite_score,
                            "disagreements": len(disagreements),
                        },
                    )
                except Exception as exc:
                    logger.warning("redis_bus_publish_failed", round=round_num, error=str(exc))
            if self.mlflow_tracker is not None:
                try:
                    self.mlflow_tracker.log_round_convergence(convergence_signal)
                except Exception as exc:
                    logger.warning("mlflow_tracker_log_round_failed", round=round_num, error=str(exc))

            # Directives for the *next* round are built from mode + this
            # round's outputs (e.g. explore mode's rotating "argue against
            # majority" injection) so they land on round_num + 1's agents.
            directives = self.controller.build_directives(mode, agent_outputs, round=round_num)
            prior_round_outputs = agent_outputs

            # Post-round budget check: allocate()'s pre-round guard (raising
            # BudgetExhaustedError) is necessarily lazy — it only fires on the
            # *next* round's call, by which point this round's writes/emits
            # have already happened and the loop would otherwise plow into
            # another allocate() call that's certain to raise. Checking here,
            # right after this round's real usage is known, lets the debate
            # stop cleanly with a partial trace (as many rounds as actually
            # completed) instead of crashing mid-iteration on the next one.
            if budget_manager.remaining_budget() < 0 and round_num < self.config.num_rounds:
                logger.warning(
                    "budget_exhausted_after_round",
                    round=round_num,
                    remaining_budget=budget_manager.remaining_budget(),
                )
                await self._emit(
                    {
                        "event": "budget_exhausted",
                        "run_id": run_id,
                        "round": round_num,
                        "remaining_budget": budget_manager.remaining_budget(),
                    }
                )
                break

        # Real edge case found via a live debate: allocate()'s viable-
        # allocation floor (budget_manager.py's MIN_VIABLE_ALLOCATION) can
        # now raise on round 1 itself for a small enough total_token_budget,
        # not only on a later round after prior overspend — leaving
        # trace.rounds empty, which trace.rounds[-1] can't handle. Route
        # straight to synthesize()'s own "no agent outputs" handling
        # (_clean_consensus_memo already produces a legitimate degraded
        # PASS memo for this — the same path an all-agents-excluded final
        # round already takes) rather than crash on the empty-list lookup.
        final_round = trace.rounds[-1] if trace.rounds else None

        async def spawn_agent_fn(
            opposing_outputs: list[AgentOutput], contested_factors: list[str]
        ) -> AgentOutput:
            if self.budget_gate is None:
                raise ValueError("tie_breaker strategy requires a budget_gate on the orchestrator")
            tie_breaker_agent = TieBreakerAgent(budget_gate=self.budget_gate)
            output = await tie_breaker_agent.resolve(
                opposing_outputs, contested_factors, max_tokens=budget_manager.remaining_reserve()
            )
            budget_manager.draw_from_reserve(output.tokens_used)
            logger.info(
                "tie_breaker_spawned",
                stance=output.stance.value,
                confidence=output.confidence,
                tokens_used=output.tokens_used,
            )
            return output

        strategy = build_strategy(self.config.conflict_resolution_strategy)
        synthesis_memo, resolved_disagreements = await synthesize(
            final_round_outputs=final_round.agent_outputs if final_round is not None else [],
            final_round_disagreements=final_round.disagreements if final_round is not None else [],
            strategy=strategy,
            remaining_budget=budget_manager.remaining_reserve(),
            spawn_agent_fn=spawn_agent_fn,
            convergence_types=(
                final_round.convergence_signal.convergence_types if final_round is not None else {}
            ),
        )
        trace.synthesis = synthesis_memo
        if resolved_disagreements and final_round is not None:
            # Replace the final round's disagreement records (and the
            # top-level mirror) with their resolved versions.
            final_round.disagreements = resolved_disagreements
            trace.disagreements = trace.disagreements[: -len(resolved_disagreements)] + resolved_disagreements

        trace.budget_ledger = budget_manager.ledger
        trace.ended_at = datetime.now(timezone.utc)
        trace.total_tokens_used = sum(entry.tokens_used for entry in trace.budget_ledger)

        if self.trace_store is not None:
            await self.trace_store.save_final(run_id, trace)
            if hasattr(self.trace_store, "save_checkpoint"):
                await self.trace_store.save_checkpoint(
                    DebateCheckpoint(
                        run_id=run_id,
                        last_completed_round=len(trace.rounds),
                        phase="complete",
                        # Same reasoning as the per-round checkpoint above:
                        # read from budget_gate when present so this agrees
                        # with debate_budgets.remaining rather than reporting
                        # BudgetManager's smaller spendable-only slice.
                        budget_remaining=(
                            self.budget_gate.remaining
                            if self.budget_gate is not None
                            else budget_manager.remaining_budget()
                        ),
                        updated_at=trace.ended_at,
                    )
                )
        if self.redis_bus is not None:
            try:
                await self.redis_bus.publish_round_event(
                    run_id,
                    {
                        "event": "synthesis_complete",
                        "recommendation": synthesis_memo.recommendation.value,
                    },
                )
            except Exception as exc:
                logger.warning(
                    "redis_bus_publish_failed", published_event="synthesis_complete", error=str(exc)
                )
        if self.mlflow_tracker is not None:
            try:
                self.mlflow_tracker.log_final(trace)
            except Exception as exc:
                logger.warning("mlflow_tracker_log_final_failed", error=str(exc))

        await self._emit(
            {
                "event": "synthesis_complete",
                "run_id": run_id,
                "recommendation": synthesis_memo.recommendation.value,
                "confidence": synthesis_memo.confidence,
            }
        )

        duration = time.monotonic() - start_time
        debate_duration_seconds.observe(duration)
        logger.info(
            "debate_completed",
            duration_seconds=duration,
            total_tokens_used=trace.total_tokens_used,
            recommendation=synthesis_memo.recommendation.value,
        )

        self._clear_run_gauges(run_id)

        return trace

    def _clear_run_gauges(self, run_id: str) -> None:
        """Prometheus Gauges hold their last-set value forever — nothing
        about a debate finishing tells them to stop reporting it. Left
        alone, every run_id this process ever handled accumulates in
        /metrics and in Grafana's legend indefinitely (observed live: a
        debate that finished minutes earlier was still showing a flat
        line and "0 active" forever, on every subsequent scrape, because
        no one had ever called .remove() on its labels). Real accumulation
        only matters on a long-lived server handling many debates, but
        there's no reason not to clean up the moment a run's numbers stop
        being current. Best-effort: prometheus_client's Gauge.remove() is a
        silent no-op for a label combination that was never set (verified
        directly against the installed version), but that isn't a
        documented cross-version guarantee — the try/except is defensive
        in depth so a `KeyError` from some other version can never fail a
        debate that already completed successfully."""
        try:
            debate_convergence_score.remove(run_id)
        except KeyError:
            pass
        for agent in self.agents:
            try:
                debate_active_agent.remove(run_id, agent.agent_id)
            except KeyError:
                pass
