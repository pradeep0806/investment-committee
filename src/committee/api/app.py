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
from committee.orchestrator_factory import build_orchestrator

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
        _stream_debate(settings, body),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _stream_debate(settings, body: DebateRequestBody):
    """SSE generator: the orchestrator's event_sink pushes onto an in-process
    queue, which this generator drains and formats as SSE frames — the same
    event hook that also (via RedisBus, wired in by build_orchestrator)
    publishes to Redis for any other consumer, so a debate running in this
    process doesn't need to round-trip through Redis to stream to its own
    request."""
    queue: asyncio.Queue = asyncio.Queue()

    async def event_sink(event: dict) -> None:
        await queue.put(event)

    orchestrator = build_orchestrator(settings, body.config)
    orchestrator.event_sink = event_sink

    async def run_and_signal_done():
        try:
            trace = await orchestrator.run(body.request)
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
