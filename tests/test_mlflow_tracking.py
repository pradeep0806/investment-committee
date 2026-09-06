from datetime import datetime, timezone

from committee.models.requests import DebateConfig, ThesisRequest
from committee.models.trace import DebateTrace
from committee.observability.mlflow_tracking import MlflowRunTracker


def _trace() -> DebateTrace:
    now = datetime.now(timezone.utc)
    return DebateTrace(
        run_id="test-run",
        request=ThesisRequest(thesis="Test thesis"),
        config=DebateConfig(),
        started_at=now,
        ended_at=now,
        total_tokens_used=100,
    )


def test_log_final_ends_run_even_when_artifact_logging_fails(monkeypatch):
    """Real bug found via live testing: with a local-path
    --default-artifact-root, mlflow.log_text() writes artifact bytes
    directly to that path rather than through the tracking server's HTTP
    API. When the calling process doesn't share the server's filesystem
    (e.g. the api container vs. the mlflow container in docker-compose,
    running as different users with no shared /mlruns mount), this raises
    PermissionError — which used to be caught by a single broad
    try/except wrapping the whole method, so mlflow.end_run() was never
    reached. The run then stayed "Running" forever in the MLflow UI even
    though its metrics had already been logged successfully. end_run()
    must fire regardless of whether artifact logging succeeds."""
    tracker = MlflowRunTracker(tracking_uri="sqlite:///:memory:")
    tracker._active_run = object()  # bypass start_run(); only log_final is under test

    import committee.observability.mlflow_tracking as module

    monkeypatch.setattr(module.mlflow, "log_metrics", lambda *a, **kw: None)

    def _raise_permission_denied(*args, **kwargs):
        raise PermissionError("[Errno 13] Permission denied: '/mlruns'")

    monkeypatch.setattr(module.mlflow, "log_text", _raise_permission_denied)

    end_run_calls = []
    monkeypatch.setattr(module.mlflow, "end_run", lambda: end_run_calls.append(1))

    tracker.log_final(_trace())

    assert end_run_calls == [1]
    assert tracker._active_run is None


def test_log_final_ends_run_when_everything_succeeds(monkeypatch):
    tracker = MlflowRunTracker(tracking_uri="sqlite:///:memory:")
    tracker._active_run = object()

    import committee.observability.mlflow_tracking as module

    logged_metrics = {}
    monkeypatch.setattr(module.mlflow, "log_metrics", lambda metrics: logged_metrics.update(metrics))
    monkeypatch.setattr(module.mlflow, "log_text", lambda *a, **kw: None)
    end_run_calls = []
    monkeypatch.setattr(module.mlflow, "end_run", lambda: end_run_calls.append(1))

    tracker.log_final(_trace())

    assert logged_metrics["tokens_used"] == 100
    assert end_run_calls == [1]
    assert tracker._active_run is None


def test_log_final_ends_run_even_when_metrics_logging_also_fails(monkeypatch):
    """Metrics logging is a separate failure domain from artifacts — a
    broken tracking connection shouldn't leave a dangling active run
    either, since start_run() already happened and there's nothing further
    to gain from keeping it open."""
    tracker = MlflowRunTracker(tracking_uri="sqlite:///:memory:")
    tracker._active_run = object()

    import committee.observability.mlflow_tracking as module

    def _raise(*args, **kwargs):
        raise ConnectionError("tracking server unreachable")

    monkeypatch.setattr(module.mlflow, "log_metrics", _raise)
    monkeypatch.setattr(module.mlflow, "log_text", _raise)
    end_run_calls = []
    monkeypatch.setattr(module.mlflow, "end_run", lambda: end_run_calls.append(1))

    tracker.log_final(_trace())

    assert end_run_calls == [1]
    assert tracker._active_run is None


def test_log_final_is_a_noop_when_not_enabled():
    tracker = MlflowRunTracker(tracking_uri="sqlite:///:memory:")
    tracker._enabled = False
    tracker._active_run = object()

    tracker.log_final(_trace())  # must not raise despite mlflow not being mocked

    assert tracker._active_run is not None  # untouched — log_final returned early
