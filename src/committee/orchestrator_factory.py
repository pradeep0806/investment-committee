"""Constructs a DebateOrchestrator from Settings + a DebateConfig.

CLI (cli/main.py) and API (api/app.py, step 9) both call this rather than
each re-implementing agent/client construction — keeps orchestrator.py itself
free of config-reading or backend-selection logic, per the plan's dependency-
injection decision (storage/LLM backends are constructed by the caller, not
self-constructed inside the orchestrator).
"""

from __future__ import annotations

import structlog

from committee.agents.registry import build_agents
from committee.config import Settings
from committee.llm.client import build_llm_client_from_settings
from committee.models.requests import DebateConfig
from committee.orchestration.budget_gate import BudgetGate
from committee.orchestration.explore_exploit import ExploreExploitController
from committee.orchestration.orchestrator import DebateOrchestrator
from committee.storage.best_effort import BestEffortTraceStore
from committee.storage.json_store import JsonStore

logger = structlog.get_logger()


def build_orchestrator(
    settings: Settings, config: DebateConfig, enable_mongo: bool = True, enable_redis: bool = True
) -> DebateOrchestrator:
    """Constructs a fully-wired DebateOrchestrator: JsonStore always (the
    source of truth), Mongo/Redis best-effort (wrapped so a connection
    failure at construction time doesn't block a debate that never needed
    them to succeed), and an MLflow run tracker. CLI and API both call this.

    `config`'s optional llm_provider/llm_model/llm_temperature/
    llm_thinking_budget override the server's .env defaults for this one
    debate's LLMClient — None (the default) means "use whatever .env says,"
    same as before this override path existed. convergence_low_threshold/
    convergence_high_threshold follow the same None-means-server-default
    pattern, falling back to Settings.convergence_low_threshold/
    convergence_high_threshold.
    """
    llm_client = build_llm_client_from_settings(
        settings,
        provider=config.llm_provider,
        model=config.llm_model,
        temperature=config.llm_temperature,
        thinking_budget=config.llm_thinking_budget,
    )
    # BudgetGate is the sole holder of llm_client from here on — agents and
    # the tie-breaker only ever receive the gate (see budget_gate.py), never
    # this reference, so there is no code path left that can reach the LLM
    # without going through budget enforcement.
    budget_gate = BudgetGate(llm_client=llm_client, total_budget=config.total_token_budget)
    agents = build_agents(budget_gate=budget_gate, agent_roles=config.agent_roles)

    # budget_store, when Mongo is enabled, backs the gate with an atomic
    # $inc-based ledger scoped by run_id — the gate binds to a specific
    # run_id once the orchestrator resolves one (see DebateOrchestrator.run
    # and BudgetGate.bind_run), since no run_id exists yet at this point.
    budget_store = None
    if enable_mongo:
        try:
            from committee.storage.budget_store import BudgetStore

            budget_store = BudgetStore(mongo_uri=settings.mongo_uri, mongo_db=settings.mongo_db)
        except Exception as exc:
            logger.warning("budget_store_unavailable", error=str(exc))
    controller = ExploreExploitController(
        low_threshold=config.convergence_low_threshold
        if config.convergence_low_threshold is not None
        else settings.convergence_low_threshold,
        high_threshold=config.convergence_high_threshold
        if config.convergence_high_threshold is not None
        else settings.convergence_high_threshold,
    )

    json_store = JsonStore(trace_json_dir=settings.trace_json_dir)
    trace_store = json_store
    if enable_mongo:
        try:
            from committee.storage.mongo_store import MongoStore

            mongo_store = MongoStore(mongo_uri=settings.mongo_uri, mongo_db=settings.mongo_db)
            trace_store = _DualTraceStore(json_store, BestEffortTraceStore(mongo_store, "mongo"))
        except Exception as exc:
            logger.warning("mongo_store_unavailable", error=str(exc))

    redis_bus = None
    if enable_redis:
        try:
            from committee.storage.redis_bus import RedisBus

            redis_bus = RedisBus(redis_url=settings.redis_url)
        except Exception as exc:
            logger.warning("redis_bus_unavailable", error=str(exc))

    mlflow_tracker = None
    try:
        from committee.observability.mlflow_tracking import MlflowRunTracker

        mlflow_tracker = MlflowRunTracker(tracking_uri=settings.mlflow_tracking_uri)
    except Exception as exc:
        logger.warning("mlflow_tracker_unavailable", error=str(exc))

    return DebateOrchestrator(
        config=config,
        agents=agents,
        controller=controller,
        budget_gate=budget_gate,
        budget_store=budget_store,
        trace_store=trace_store,
        redis_bus=redis_bus,
        mlflow_tracker=mlflow_tracker,
    )


class _DualTraceStore:
    """save_round/save_final fan out to JSON (always, blocking, source of
    truth — its exceptions propagate) and the best-effort backend (already
    self-swallowing). get_run/list_runs read from JSON only, since it's the
    only store guaranteed to hold a complete, consistent picture."""

    def __init__(self, json_store: JsonStore, best_effort_store: BestEffortTraceStore):
        self._json_store = json_store
        self._best_effort_store = best_effort_store

    async def save_round(self, run_id, round_record) -> None:
        await self._json_store.save_round(run_id, round_record)
        await self._best_effort_store.save_round(run_id, round_record)

    async def save_final(self, run_id, trace) -> None:
        await self._json_store.save_final(run_id, trace)
        await self._best_effort_store.save_final(run_id, trace)

    async def get_run(self, run_id):
        return await self._json_store.get_run(run_id)

    async def list_runs(self):
        return await self._json_store.list_runs()

    async def save_checkpoint(self, checkpoint) -> None:
        await self._json_store.save_checkpoint(checkpoint)
        await self._best_effort_store.save_checkpoint(checkpoint)

    async def get_checkpoint(self, run_id):
        return await self._json_store.get_checkpoint(run_id)

    def register_trace(self, run_id, trace) -> None:
        self._json_store.register_trace(run_id, trace)
