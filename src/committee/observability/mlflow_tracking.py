"""MLflow tracking: one run per debate (CLAUDE.md §8). Params logged at
start; the convergence-score-per-round series logged as metric history
round-by-round (so a completed run's explore->exploit trajectory is
inspectable after the fact, not just live in Grafana); final
metrics/artifacts logged at save_final.

MLflow isn't named in CLAUDE.md's explicit "best-effort" list (that's Mongo/
Redis specifically), but the same principle applies: it's an observability
sidecar, not core debate logic, so a broken or misconfigured tracking backend
(e.g. a filestore incompatibility between MLflow versions) must never crash
a debate. Every public method here is defensive internally — logged and
swallowed on failure — the same pattern RedisBus already uses.
"""

from __future__ import annotations

import mlflow
import structlog

from committee.models.requests import DebateConfig, ThesisRequest
from committee.models.synthesis import ConvergenceSignal
from committee.models.trace import DebateTrace

logger = structlog.get_logger()


class MlflowRunTracker:
    def __init__(self, tracking_uri: str):
        self._enabled = True
        try:
            mlflow.set_tracking_uri(tracking_uri)
        except Exception as exc:
            logger.warning("mlflow_set_tracking_uri_failed", error=str(exc))
            self._enabled = False
        self._active_run = None

    def start_run(self, request: ThesisRequest, config: DebateConfig, model: str, num_agents: int) -> None:
        if not self._enabled:
            return
        try:
            self._active_run = mlflow.start_run()
            mlflow.log_params(
                {
                    "thesis": request.thesis[:250],  # MLflow param values are length-capped
                    "budget": config.total_token_budget,
                    "model": model,
                    "num_agents": num_agents,
                    "strategy": config.conflict_resolution_strategy,
                }
            )
        except Exception as exc:
            logger.warning("mlflow_start_run_failed", error=str(exc))
            self._enabled = False
            self._active_run = None

    def log_round_convergence(self, signal: ConvergenceSignal) -> None:
        if not self._enabled or self._active_run is None:
            return
        try:
            mlflow.log_metric("convergence_score", signal.composite_score, step=signal.round)
        except Exception as exc:
            logger.warning("mlflow_log_round_convergence_failed", error=str(exc))

    def log_final(self, trace: DebateTrace) -> None:
        if not self._enabled or self._active_run is None:
            return
        try:
            mlflow.log_metrics(
                {
                    "tokens_used": trace.total_tokens_used,
                    "rounds_run": len(trace.rounds),
                    "disagreements_count": len(trace.disagreements),
                    "convergence_at_final_round": trace.convergence_signals[-1].composite_score
                    if trace.convergence_signals
                    else 0.0,
                }
            )
        except Exception as exc:
            logger.warning("mlflow_log_final_metrics_failed", error=str(exc))

        # Real bug found via live testing: with a local-path
        # --default-artifact-root, the MLflow client writes artifact bytes
        # directly to that path rather than proxying them through the
        # tracking server's HTTP API — and the api container doesn't share
        # the mlflow container's filesystem, so this reliably fails with
        # "Permission denied". Logging artifacts in their own try/except,
        # after metrics and before end_run(), means that failure can no
        # longer leave a run stuck "Running" forever with metrics recorded
        # but never closed — the full trace is already the JSON file under
        # TRACE_JSON_DIR regardless, so a failed trace.json/synthesis.json
        # artifact upload here is a pure nice-to-have, not the source of
        # truth.
        try:
            mlflow.log_text(trace.model_dump_json(indent=2), "trace.json")
            if trace.synthesis:
                mlflow.log_text(trace.synthesis.model_dump_json(indent=2), "synthesis.json")
        except Exception as exc:
            logger.warning("mlflow_log_final_artifacts_failed", error=str(exc))

        try:
            mlflow.end_run()
        except Exception as exc:
            logger.warning("mlflow_end_run_failed", error=str(exc))
        finally:
            self._active_run = None
