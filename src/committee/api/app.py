"""FastAPI layer: /health, /metrics (infra, don't count against "one
endpoint" per CLAUDE.md §1.2), and the one business endpoint POST /debate,
with `?stream=true` returning the same computation as Server-Sent Events on
the same route.

Note for any client: browsers' native EventSource only supports GET, so a
streaming client must use fetch() with a ReadableStream reader, not
EventSource — this is a POST route.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator

from committee.api.schemas import DebateRequestBody, HealthResponse
from committee.config import get_settings
from committee.observability.logging import configure_logging
from committee.observability.metrics import REGISTRY
from committee.orchestration.budget_manager import BudgetExhaustedError
from committee.orchestration.orchestrator import RunAlreadyInProgressError
from committee.orchestrator_factory import build_orchestrator
from committee.storage.json_store import JsonStore

app = FastAPI(title="The Investment Committee", version="0.1.0")

_settings = get_settings()
configure_logging(log_level=_settings.log_level, log_format=_settings.log_format)

# Permissive CORS — this is a local dev/demo viewer (frontend/), not a
# public-facing deployment, so an allowlist of one origin isn't worth the
# maintenance; tighten this before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ordinary HTTP request metrics (latency, status codes, in-progress requests)
# — generic infra observability, not domain-specific, but cheap to add
# alongside the custom debate metrics already exposed at /metrics.
Instrumentator().instrument(app)


@app.get("/health")
async def health() -> HealthResponse:
    return HealthResponse()


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.post("/debate")
async def debate(body: DebateRequestBody, stream: bool = False):
    settings = get_settings()

    if not stream:
        orchestrator = build_orchestrator(settings, body.config)
        try:
            trace = await orchestrator.run(body.request)
        except BudgetExhaustedError as exc:
            # A genuine out-of-budget event (BudgetManager's dynamic
            # rebaselining already absorbs ordinary over/under-spends across
            # rounds — this only fires when there's truly nothing left even
            # for a 1-token floor per agent). 422: the request was
            # well-formed, but total_token_budget was too small for what the
            # LLM actually needed. Never leak the raw traceback to a client.
            raise HTTPException(status_code=422, detail=f"Debate budget exhausted: {exc}")
        return trace.model_dump(mode="json")

    return StreamingResponse(
        _stream_run(settings, body.config, lambda orchestrator: orchestrator.run(body.request)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/debate/{run_id}/resume")
async def resume_debate(run_id: str, stream: bool = False):
    """Additive endpoint, not a change to POST /debate's contract — resumes
    an incomplete debate from its last completed round (CLAUDE.md hardening
    task, Phase C), using the request/config it was originally started with
    (read back from its saved trace, the source of truth) rather than
    requiring the caller to resend them. Mirrors `committee resume` in the
    CLI, which reads the same JsonStore path.
    """
    settings = get_settings()
    trace = await _load_resumable_trace(settings, run_id)

    def run_resumed(orchestrator):
        return orchestrator.run(trace.request, run_id=run_id, resume=True)

    if not stream:
        orchestrator = build_orchestrator(settings, trace.config)
        try:
            resumed_trace = await run_resumed(orchestrator)
        except BudgetExhaustedError as exc:
            raise HTTPException(status_code=422, detail=f"Debate budget exhausted: {exc}")
        except RunAlreadyInProgressError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return resumed_trace.model_dump(mode="json")

    return StreamingResponse(
        _stream_run(settings, trace.config, run_resumed),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _load_resumable_trace(settings, run_id: str):
    """Reads the saved trace for run_id and confirms it's actually something
    to resume (exists, not already complete) — a 404/409 here beats a raw
    orchestrator crash from resume() silently falling back to a fresh,
    differently-configured start."""
    json_store = JsonStore(trace_json_dir=settings.trace_json_dir)
    trace = await json_store.get_run(run_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"No saved trace found for run_id={run_id!r}")
    if trace.ended_at is not None:
        raise HTTPException(status_code=409, detail=f"run_id={run_id!r} has already completed")
    return trace


async def _stream_run(settings, config, run_fn):
    """SSE generator shared by POST /debate?stream=true and
    POST /debate/{run_id}/resume?stream=true: the orchestrator's event_sink
    pushes onto an in-process queue, which this generator drains and formats
    as SSE frames — the same event hook that also (via RedisBus, wired in by
    build_orchestrator) publishes to Redis for any other consumer, so a
    debate running in this process doesn't need to round-trip through Redis
    to stream to its own request. `run_fn(orchestrator)` is the one thing
    that differs between a fresh start and a resume — everything else about
    driving the SSE stream is identical."""
    queue: asyncio.Queue = asyncio.Queue()

    async def event_sink(event: dict) -> None:
        await queue.put(event)

    orchestrator = build_orchestrator(settings, config)
    orchestrator.event_sink = event_sink

    async def run_and_signal_done():
        try:
            trace = await run_fn(orchestrator)
            await queue.put({"event": "done", "trace": trace.model_dump(mode="json")})
        except Exception as exc:
            await queue.put({"event": "error", "message": str(exc)})

    run_task = asyncio.create_task(run_and_signal_done())

    try:
        while True:
            event = await queue.get()
            event_type = event.get("event", "message")
            yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n"
            if event_type in ("done", "error"):
                break
    finally:
        if not run_task.done():
            run_task.cancel()
