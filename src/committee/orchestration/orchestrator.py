"""DebateOrchestrator: the core loop, with no web-framework dependency (CLAUDE.md §1.1).

CLI and API both call into this same plain Python object. As of step 5, the
loop runs the configured `config.num_rounds` (not a hardcoded 2), and mode
each round is a *computed* signal from ExploreExploitController — never a
hardcoded round number (CLAUDE.md §1.6). Budget reallocation in exploit mode
targets the controller's contested agents; explore mode injects the
controller's rotating "argue against majority" directive.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from typing import Any, Awaitable, Callable

from committee.agents.base import AnalystAgent
from committee.agents.tie_breaker import TieBreakerAgent
from committee.llm.client import LLMClient
from committee.llm.structured_output import LLMValidationError
from committee.models.agent_output import AgentOutput
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
from committee.orchestration.budget_manager import BudgetManager
from committee.orchestration.conflict_resolution.registry import build_strategy
from committee.orchestration.disagreement import detect as detect_disagreements
from committee.orchestration.explore_exploit import ExploreExploitController
from committee.synthesis.synthesizer import synthesize

EventSink = Callable[[dict], Awaitable[None]]


class DebateOrchestrator:
    def __init__(
        self,
        config: DebateConfig,
        agents: list[AnalystAgent],
        controller: ExploreExploitController | None = None,
        llm_client: LLMClient | None = None,
        trace_store: Any | None = None,
        redis_bus: Any | None = None,
        mlflow_tracker: Any | None = None,
        event_sink: EventSink | None = None,
    ):
        self.config = config
        self.agents = agents
        self.controller = controller or ExploreExploitController(
            low_threshold=config.convergence_low_threshold,
            high_threshold=config.convergence_high_threshold,
        )
        # Only needed if conflict_resolution_strategy=tie_breaker actually
        # ends up spawning an agent; every other strategy ignores it.
        self.llm_client = llm_client
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

    async def run(self, request: ThesisRequest) -> DebateTrace:
        run_id = str(uuid.uuid4())
        logger = get_run_logger(run_id)
        started_at = datetime.now(timezone.utc)

        trace = DebateTrace(
            run_id=run_id,
            request=request,
            config=self.config,
            started_at=started_at,
        )

        budget_manager = BudgetManager(
            total_token_budget=self.config.total_token_budget,
            num_rounds=self.config.num_rounds,
            num_agents=len(self.agents),
        )

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
                    model=self.llm_client.model if self.llm_client else "unknown",
                    num_agents=len(self.agents),
                )
            except Exception as exc:
                logger.warning("mlflow_tracker_start_run_failed", error=str(exc))

        start_time = time.monotonic()
        logger.info("debate_started", thesis=request.thesis, num_agents=len(self.agents))

        prior_round_outputs: list[AgentOutput] | None = None
        directives: dict[str, str] = {}
        previous_mode: str | None = None

        for round_num in range(1, self.config.num_rounds + 1):
            round_started_at = datetime.now(timezone.utc)

            # Round 1 has no prior outputs to score convergence on, so it
            # always starts in explore mode — there's nothing yet to converge
            # around. From round 2 onward, mode is decided from the *previous*
            # round's computed convergence signal.
            if prior_round_outputs is None:
                mode = "explore"
                contested = []
            else:
                signal = self.controller.score(prior_round_outputs, round=round_num - 1)
                mode = self.controller.decide_mode(signal)
                contested = self.controller.contested_agents(prior_round_outputs)

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

            allocations = budget_manager.allocate(
                round=round_num,
                agent_ids=[agent.agent_id for agent in self.agents],
                mode=mode,
                contested_agents=contested,
            )

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
                    continue

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
                    }
                )
                agent_outputs.append(output)
                budget_manager.record_actual_usage(
                    round=round_num,
                    agent_id=agent.agent_id,
                    tokens_allocated=token_budget,
                    tokens_used=output.tokens_used,
                    mode=mode,
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

            convergence_signal = self.controller.score(agent_outputs, round=round_num)
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

        final_round = trace.rounds[-1]

        async def spawn_agent_fn(
            opposing_outputs: list[AgentOutput], contested_factors: list[str]
        ) -> AgentOutput:
            if self.llm_client is None:
                raise ValueError("tie_breaker strategy requires an llm_client on the orchestrator")
            tie_breaker_agent = TieBreakerAgent(llm_client=self.llm_client)
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
            final_round_outputs=final_round.agent_outputs,
            final_round_disagreements=final_round.disagreements,
            strategy=strategy,
            remaining_budget=budget_manager.remaining_reserve(),
            spawn_agent_fn=spawn_agent_fn,
        )
        trace.synthesis = synthesis_memo
        if resolved_disagreements:
            # Replace the final round's disagreement records (and the
            # top-level mirror) with their resolved versions.
            final_round.disagreements = resolved_disagreements
            trace.disagreements = trace.disagreements[: -len(resolved_disagreements)] + resolved_disagreements

        trace.budget_ledger = budget_manager.ledger
        trace.ended_at = datetime.now(timezone.utc)
        trace.total_tokens_used = sum(entry.tokens_used for entry in trace.budget_ledger)

        if self.trace_store is not None:
            await self.trace_store.save_final(run_id, trace)
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

        return trace
