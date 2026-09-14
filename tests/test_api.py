import json

import pytest
from httpx import ASGITransport, AsyncClient

from committee.api.app import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


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


class _HugeUsageRawCaller:
    """Every call reports using far more tokens than any reasonable budget
    allocation — simulates the real bug caught in docker-compose smoke
    testing, where round 1 alone exhausted the entire spendable budget."""

    async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens=None):
        return (
            {
                "stance": "Buy",
                "confidence": 65,
                "key_factors": ["growth"],
                "evidence": ["Q3 revenue up 22% YoY"],
                "top_risk": "x",
            },
            50_000,
        )


@pytest.fixture
def fake_orchestrator_factory(monkeypatch):
    """Patches build_orchestrator (imported into api.app's namespace) so API
    tests exercise real FastAPI routing/serialization without a real LLM call
    or real Mongo/Redis/MLflow connections."""
    from committee.agents.registry import build_agents
    from committee.llm.client import LLMClient
    from committee.orchestration.budget_gate import BudgetGate
    from committee.orchestration.orchestrator import DebateOrchestrator

    def _build(settings, config, **kwargs):
        client = LLMClient.__new__(LLMClient)
        client.provider = "fake"
        client.model = "fake-model"
        client.api_key = "fake-key"
        client.timeout_seconds = 60
        client.max_retries = 3
        client._raw_caller = _FakeRawCaller()

        gate = BudgetGate(llm_client=client, total_budget=config.total_token_budget)
        agents = build_agents(budget_gate=gate)
        return DebateOrchestrator(config=config, agents=agents, budget_gate=gate)

    import committee.api.app as app_module

    monkeypatch.setattr(app_module, "build_orchestrator", _build)
    return _build


async def test_health_returns_ok():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_metrics_returns_prometheus_text():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics")
    assert response.status_code == 200
    assert "debate_duration_seconds" in response.text


async def test_post_debate_non_streaming_returns_full_trace(fake_orchestrator_factory):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/debate",
            json={
                "request": {"thesis": "Test thesis for API"},
                "config": {"total_token_budget": 8000, "num_rounds": 2},
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["run_id"]
    assert len(body["rounds"]) == 2
    assert body["synthesis"]["recommendation"] == "Buy"


async def test_post_debate_stream_returns_sse_events(fake_orchestrator_factory):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/debate?stream=true",
            json={
                "request": {"thesis": "Test thesis for streaming"},
                "config": {"total_token_budget": 8000, "num_rounds": 2},
            },
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")

            raw_events = []
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    frame, buffer = buffer.split("\n\n", 1)
                    raw_events.append(frame)

    event_types = []
    for frame in raw_events:
        lines = frame.split("\n")
        event_line = next((l for l in lines if l.startswith("event: ")), None)
        data_line = next((l for l in lines if l.startswith("data: ")), None)
        if event_line and data_line:
            event_types.append(event_line[len("event: ") :])
            json.loads(data_line[len("data: ") :])  # each data payload is valid JSON

    assert "round_start" in event_types
    assert "agent_reasoning_start" in event_types
    assert "agent_reasoning_end" in event_types
    assert "convergence_computed" in event_types
    assert event_types[-1] == "done"


async def test_post_debate_non_streaming_stops_early_and_succeeds_when_a_round_exhausts_budget(
    monkeypatch,
):
    """A round's real usage can now exceed the entire spendable budget
    without crashing the debate: the orchestrator's post-round check (added
    alongside the tokens_allocated cap fix) stops early with a partial trace
    rather than letting the next round's allocate() call raise
    BudgetExhaustedError. This replaces an earlier version of this test that
    asserted a raw 500 (before the 422 handler existed) — now, thanks to the
    early-stop, this fixture doesn't even reach BudgetExhaustedError at all;
    see the test below for the case that genuinely still raises it."""
    from committee.agents.registry import build_agents
    from committee.llm.client import LLMClient
    from committee.orchestration.budget_gate import BudgetGate
    from committee.orchestration.orchestrator import DebateOrchestrator

    def _build(settings, config, **kwargs):
        client = LLMClient.__new__(LLMClient)
        client.provider = "fake"
        client.model = "fake-model"
        client.api_key = "fake-key"
        client.timeout_seconds = 60
        client.max_retries = 3
        client._raw_caller = _HugeUsageRawCaller()

        gate = BudgetGate(llm_client=client, total_budget=config.total_token_budget)
        agents = build_agents(budget_gate=gate)
        return DebateOrchestrator(config=config, agents=agents, budget_gate=gate)

    import committee.api.app as app_module

    monkeypatch.setattr(app_module, "build_orchestrator", _build)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/debate",
            json={
                "request": {"thesis": "Test thesis for budget exhaustion"},
                "config": {"total_token_budget": 1000, "num_rounds": 2},
            },
        )

    assert response.status_code == 200
    body = response.json()
    # Only round 1 completed — round 2 never started, since the post-round
    # check caught the overrun before allocate() would have raised for it.
    assert len(body["rounds"]) == 1


async def test_post_debate_non_streaming_returns_422_when_a_single_round_cannot_allocate_a_floor(
    monkeypatch,
):
    """The genuine BudgetExhaustedError path that survives the early-stop
    fix: a budget so small that even round 1's very first allocate() call
    can't cover a floor allocation for every agent. Must be a clean 422, not
    a leaked raw 500 traceback — the original bug this pair of tests
    guarded against."""
    from committee.agents.registry import build_agents
    from committee.llm.client import LLMClient
    from committee.orchestration.budget_gate import BudgetGate
    from committee.orchestration.orchestrator import DebateOrchestrator

    def _build(settings, config, **kwargs):
        client = LLMClient.__new__(LLMClient)
        client.provider = "fake"
        client.model = "fake-model"
        client.api_key = "fake-key"
        client.timeout_seconds = 60
        client.max_retries = 3
        client._raw_caller = _HugeUsageRawCaller()

        gate = BudgetGate(llm_client=client, total_budget=config.total_token_budget)
        agents = build_agents(budget_gate=gate)
        return DebateOrchestrator(config=config, agents=agents, budget_gate=gate)

    import committee.api.app as app_module

    monkeypatch.setattr(app_module, "build_orchestrator", _build)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/debate",
            json={
                "request": {"thesis": "Test thesis for budget exhaustion"},
                # Spendable budget (90% of 2) is smaller than the 4 agents
                # needing a floor allocation each, so round 1's very first
                # allocate() call raises before any agent runs at all — the
                # post-round early-stop check never gets a chance to fire,
                # since there's no completed round for it to run after.
                "config": {"total_token_budget": 2, "num_rounds": 2},
            },
        )

    assert response.status_code == 422
    assert "budget" in response.json()["detail"].lower()


async def test_post_debate_stream_completes_when_a_round_exhausts_budget(monkeypatch):
    """Streaming counterpart of the early-stop test above: a round exhausting
    budget now ends the stream with `done` (carrying a partial trace), not
    an `error` event — the orchestrator itself never raises in this case."""
    from committee.agents.registry import build_agents
    from committee.llm.client import LLMClient
    from committee.orchestration.budget_gate import BudgetGate
    from committee.orchestration.orchestrator import DebateOrchestrator

    def _build(settings, config, **kwargs):
        client = LLMClient.__new__(LLMClient)
        client.provider = "fake"
        client.model = "fake-model"
        client.api_key = "fake-key"
        client.timeout_seconds = 60
        client.max_retries = 3
        client._raw_caller = _HugeUsageRawCaller()

        gate = BudgetGate(llm_client=client, total_budget=config.total_token_budget)
        agents = build_agents(budget_gate=gate)
        return DebateOrchestrator(config=config, agents=agents, budget_gate=gate)

    import committee.api.app as app_module

    monkeypatch.setattr(app_module, "build_orchestrator", _build)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/debate?stream=true",
            json={
                "request": {"thesis": "Test thesis for streaming budget exhaustion"},
                "config": {"total_token_budget": 1000, "num_rounds": 2},
            },
        ) as response:
            assert response.status_code == 200
            raw_events = []
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    frame, buffer = buffer.split("\n\n", 1)
                    raw_events.append(frame)

    event_types = [
        line[len("event: ") :]
        for frame in raw_events
        for line in frame.split("\n")
        if line.startswith("event: ")
    ]
    assert "budget_exhausted" in event_types
    assert event_types[-1] == "done"


class TestResumeEndpoint:
    """POST /debate/{run_id}/resume — additive, doesn't touch POST /debate's
    contract. Mirrors the CLI's `resume` command: reads the saved trace back
    from JsonStore (trace_json_dir) rather than requiring the caller to
    resend request/config."""

    @pytest.fixture
    def json_store(self, tmp_path, monkeypatch):
        from committee.config import Settings

        settings = Settings(
            llm_provider="anthropic", llm_api_key="fake", trace_json_dir=str(tmp_path)
        )

        import committee.api.app as app_module

        monkeypatch.setattr(app_module, "get_settings", lambda: settings)

        from committee.storage.json_store import JsonStore

        return JsonStore(trace_json_dir=str(tmp_path))

    async def test_resume_unknown_run_id_returns_404(self, json_store):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/debate/does-not-exist/resume")
        assert response.status_code == 404

    async def test_resume_already_completed_run_returns_409(self, json_store):
        from datetime import UTC, datetime

        from committee.models.requests import DebateConfig, ThesisRequest
        from committee.models.trace import DebateTrace

        trace = DebateTrace(
            run_id="already-done",
            request=ThesisRequest(thesis="Test thesis"),
            config=DebateConfig(total_token_budget=8000, num_rounds=2),
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
        )
        await json_store.save_final("already-done", trace)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/debate/already-done/resume")
        assert response.status_code == 409

    async def test_resume_incomplete_run_continues_and_returns_full_trace(
        self, json_store, fake_orchestrator_factory
    ):
        from datetime import UTC, datetime

        from committee.models.requests import DebateConfig, ThesisRequest
        from committee.models.trace import DebateTrace

        trace = DebateTrace(
            run_id="resume-me",
            request=ThesisRequest(thesis="Test thesis for resume"),
            config=DebateConfig(total_token_budget=8000, num_rounds=2),
            started_at=datetime.now(UTC),
            ended_at=None,
        )
        await json_store.save_final("resume-me", trace)  # writes with ended_at=None

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/debate/resume-me/resume")

        assert response.status_code == 200
        body = response.json()
        assert body["run_id"] == "resume-me"
        assert len(body["rounds"]) == 2
        assert body["ended_at"] is not None
