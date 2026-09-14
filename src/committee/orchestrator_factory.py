"""Constructs a DebateOrchestrator from Settings + a DebateConfig.

CLI (cli/main.py) and API (api/app.py, step 9) both call this rather than
each re-implementing agent/client construction — keeps orchestrator.py itself
free of config-reading or backend-selection logic, per the plan's dependency-
injection decision (storage/LLM backends are constructed by the caller, not
self-constructed inside the orchestrator).
"""

from __future__ import annotations

import asyncio
import concurrent.futures

import structlog

from committee.agents.registry import DEFAULT_AGENT_ROLES, build_agents
from committee.config import Settings
from committee.llm.client import build_llm_client_from_settings
from committee.models.persona import AgentPersona
from committee.models.requests import DebateConfig
from committee.orchestration.budget_gate import BudgetGate
from committee.orchestration.explore_exploit import ExploreExploitController
from committee.orchestration.orchestrator import DebateOrchestrator
from committee.storage.best_effort import BestEffortTraceStore
from committee.storage.json_store import JsonStore

logger = structlog.get_logger()


def _resolve_agent_selection(
    settings: Settings, agent_ids: list[str] | None
) -> tuple[list[str], list[AgentPersona]]:
    """Resolves a debate's `agent_ids` selection into (built-in agent_roles,
    custom personas to build), per Phase 1-B: every debate includes the 4
    core lenses by default, and user-defined agents supplement rather than
    replace them.

    - `agent_ids=None` (default): core 4 + every active custom persona
      currently in PersonaStore. This mirrors registry.py's own
      `agent_roles=None -> DEFAULT_AGENT_ROLES` pattern, extended to also
      pull in whatever custom personas exist, so a debate started without
      any special configuration automatically reflects the current roster.
    - `agent_ids=[...]` explicit: exactly those ids, resolved against
      PersonaStore (built-ins are seeded rows there too, see
      storage/persona_store.py) — lets a caller run with a subset instead of
      "everything active."

    If PersonaStore/Mongo is unavailable, falls back to the core 4 only
    (never blocks a debate on a persona-store outage) — mirrors the
    best-effort treatment every other Mongo-backed piece gets in this
    factory.

    Runs its own short-lived event loop via asyncio.run rather than being
    async itself: build_orchestrator is called synchronously from 5 places
    across the CLI and API (see storage/run_lock_store.py's identical note
    on why RunLockStore.ensure_index is deferred instead of awaited here),
    some of them outside any running event loop at construction time, so
    this can't be a coroutine the caller awaits.
    """
    try:
        from committee.storage.persona_store import PersonaStore

        async def _fetch() -> tuple[list[str], list[AgentPersona]]:
            persona_store = PersonaStore(mongo_uri=settings.mongo_uri, mongo_db=settings.mongo_db)
            try:
                await persona_store.ensure_builtins_seeded()
                if agent_ids is None:
                    return list(DEFAULT_AGENT_ROLES), await persona_store.list_active_custom()

                all_personas = {persona.id: persona for persona in await persona_store.list_all()}
                builtin_roles = [aid for aid in agent_ids if aid in DEFAULT_AGENT_ROLES]
                custom_personas = [
                    all_personas[aid]
                    for aid in agent_ids
                    if aid in all_personas and not all_personas[aid].is_builtin
                ]
                return builtin_roles, custom_personas
            finally:
                await persona_store.close()

        return _run_sync(_fetch())
    except Exception as exc:
        logger.warning("persona_store_unavailable", error=str(exc))
        return list(DEFAULT_AGENT_ROLES), []


def _run_sync(coro):
    """Runs `coro` to completion regardless of whether the calling thread
    already has an event loop running.

    build_orchestrator is called from both loop-free contexts (the CLI's
    `asyncio.run(orchestrator.run(...))` hasn't started yet at construction
    time) and from inside a running loop (FastAPI's `async def debate(...)`
    handler calls it synchronously mid-coroutine) — asyncio.run() alone would
    raise "cannot be called from a running event loop" in the second case.
    When a loop is already running, the coroutine is executed on a fresh
    loop in a separate thread instead, so either caller gets a plain
    blocking call.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


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

    # agent_roles predates persona-as-data and only ever names built-in
    # registry keys — if a caller sets it explicitly, honor it exactly as
    # before (no persona resolution, no implicit custom-agent inclusion).
    # agent_ids is the new, richer selector (Phase 1-B: None -> core 4 + all
    # active custom personas; explicit list -> exactly those ids, built-in
    # or custom) and only kicks in when agent_roles was left unset.
    if config.agent_roles is not None or not enable_mongo:
        # enable_mongo=False (tests, or a deliberately Mongo-less run) has no
        # PersonaStore to resolve custom agents against — fall back to
        # whatever agent_roles says, or the core 4, same as if PersonaStore
        # were unreachable at runtime.
        agent_roles, custom_personas = config.agent_roles or list(DEFAULT_AGENT_ROLES), []
    else:
        agent_roles, custom_personas = _resolve_agent_selection(settings, config.agent_ids)

    agents = build_agents(budget_gate=budget_gate, agent_roles=agent_roles, custom_personas=custom_personas)

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

    # run_lock_store, when Mongo is enabled, is the cross-pod counterpart to
    # DebateOrchestrator's in-process _active_run_locks — see
    # storage/run_lock_store.py for why the in-process lock alone isn't
    # enough once there's more than one API pod. Its unique index is
    # created lazily on first use (RunLockStore.ensure_index(), called from
    # acquire()) rather than here: this factory is synchronous and called
    # from several places (CLI, API) not all of which have a running event
    # loop at construction time, so nothing here can be awaited.
    run_lock_store = None
    if enable_mongo:
        try:
            from committee.storage.run_lock_store import RunLockStore

            run_lock_store = RunLockStore(
                mongo_uri=settings.mongo_uri,
                mongo_db=settings.mongo_db,
                lease_seconds=settings.run_lock_lease_seconds,
            )
        except Exception as exc:
            logger.warning("run_lock_store_unavailable", error=str(exc))

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
        run_lock_store=run_lock_store,
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
