"""CLI entrypoint: accepts a thesis, budget, and configuration (CLAUDE.md
§1.1, Core requirement). Invokes the exact same DebateOrchestrator the API
uses (via orchestrator_factory) — no separate orchestration logic for the
CLI path.
"""

from __future__ import annotations

import asyncio

import typer

from committee.config import get_settings
from committee.models.requests import DebateConfig, ThesisRequest
from committee.observability.logging import configure_logging
from committee.orchestrator_factory import build_orchestrator
from committee.storage.json_store import JsonStore

app = typer.Typer(help="The Investment Committee — multi-agent debate CLI.")


@app.command()
def run(
    thesis: str = typer.Option(..., "--thesis", help="The investment thesis to debate."),
    entity: str = typer.Option(None, "--entity", help="Entity/ticker context, if any."),
    budget: int = typer.Option(
        None, "--budget", help="Total token budget for the debate (defaults to .env's DEFAULT_TOKEN_BUDGET)."
    ),
    rounds: int = typer.Option(
        None, "--rounds", help="Number of debate rounds, 2-3 (defaults to .env's DEFAULT_NUM_ROUNDS)."
    ),
    strategy: str = typer.Option(
        None,
        "--strategy",
        help="Conflict resolution strategy: flag_unresolved | confidence_weighted | tie_breaker.",
    ),
    agent_ids: str = typer.Option(
        None,
        "--agent-ids",
        help=(
            "Comma-separated persona ids to include (built-in or custom, e.g. "
            "'fundamentals,risk_contrarian,<custom-persona-id>'). Defaults to "
            "the core 4 plus every active custom persona."
        ),
    ),
) -> None:
    """Run a full debate and print the synthesis memo."""
    settings = get_settings()
    configure_logging(log_level=settings.log_level, log_format=settings.log_format)

    config_kwargs = {}
    if budget is not None:
        config_kwargs["total_token_budget"] = budget
    else:
        config_kwargs["total_token_budget"] = settings.default_token_budget
    if rounds is not None:
        config_kwargs["num_rounds"] = rounds
    else:
        config_kwargs["num_rounds"] = settings.default_num_rounds
    if strategy is not None:
        config_kwargs["conflict_resolution_strategy"] = strategy
    else:
        config_kwargs["conflict_resolution_strategy"] = settings.conflict_resolution_strategy
    config_kwargs["convergence_low_threshold"] = settings.convergence_low_threshold
    config_kwargs["convergence_high_threshold"] = settings.convergence_high_threshold
    if agent_ids is not None:
        config_kwargs["agent_ids"] = [a.strip() for a in agent_ids.split(",") if a.strip()]

    config = DebateConfig(**config_kwargs)
    request = ThesisRequest(thesis=thesis, entity=entity)

    orchestrator = build_orchestrator(settings, config)

    typer.echo(f"Running debate: {thesis!r} (budget={config.total_token_budget}, rounds={config.num_rounds})")
    trace = asyncio.run(orchestrator.run(request))

    _print_summary(trace)


def _print_summary(trace) -> None:
    typer.echo("")
    typer.echo(f"run_id: {trace.run_id}")
    typer.echo(f"total_tokens_used: {trace.total_tokens_used}")
    typer.echo(f"rounds_run: {len(trace.rounds)}")

    for round_record in trace.rounds:
        typer.echo(
            f"  round {round_record.round}: mode={round_record.convergence_signal.mode_selected} "
            f"convergence={round_record.convergence_signal.composite_score:.2f}"
        )
        for output in round_record.agent_outputs:
            typer.echo(
                f"    [{output.agent_id}] {output.stance.value} "
                f"(confidence={output.confidence}, tokens={output.tokens_used})"
            )
            if output.executive_summary:
                typer.echo(f"      -> {output.executive_summary}")

    if trace.disagreements:
        typer.echo("")
        typer.echo(f"Disagreements detected: {len(trace.disagreements)}")

    if trace.synthesis:
        typer.echo("")
        typer.echo("=== Synthesis ===")
        typer.echo(f"Recommendation: {trace.synthesis.recommendation.value}")
        typer.echo(f"Confidence: {trace.synthesis.confidence}")
        typer.echo(f"Supporting: {', '.join(trace.synthesis.supporting_agents) or 'none'}")
        typer.echo(f"Dissenting: {', '.join(trace.synthesis.dissenting_agents) or 'none'}")
        if trace.synthesis.dissent_appendix:
            typer.echo(f"Dissent: {trace.synthesis.dissent_appendix}")
        typer.echo("")
        typer.echo(f"Dissenting view: {trace.synthesis.dissenting_view_note}")
        for entry in trace.synthesis.dissenting_view:
            typer.echo(f"  [{entry.agent_id}] {entry.stance.value}: {entry.reason}")
        if trace.synthesis.agent_summaries:
            typer.echo("")
            typer.echo("At a glance:")
            for agent_id, summary in trace.synthesis.agent_summaries.items():
                typer.echo(f"  [{agent_id}] {summary}")


@app.command()
def replay(run_id: str = typer.Argument(..., help="The run_id to replay.")) -> None:
    """Replay a previously saved debate trace by run_id — reads the JSON
    trace (the source of truth) and prints the same summary `run` shows."""
    settings = get_settings()
    json_store = JsonStore(trace_json_dir=settings.trace_json_dir)

    trace = asyncio.run(json_store.get_run(run_id))
    if trace is None:
        typer.echo(f"No saved trace found for run_id={run_id!r} in {settings.trace_json_dir}")
        raise typer.Exit(code=1)

    _print_summary(trace)


@app.command()
def resume(run_id: str = typer.Argument(..., help="The run_id to resume.")) -> None:
    """Resume an incomplete debate from its last completed round, using the
    request/config it was originally started with (read back from its saved
    trace) — for a process that crashed or was killed mid-debate rather than
    a fresh `run`."""
    settings = get_settings()
    configure_logging(log_level=settings.log_level, log_format=settings.log_format)
    json_store = JsonStore(trace_json_dir=settings.trace_json_dir)

    trace = asyncio.run(json_store.get_run(run_id))
    if trace is None:
        typer.echo(f"No saved trace found for run_id={run_id!r} in {settings.trace_json_dir}")
        raise typer.Exit(code=1)
    if trace.ended_at is not None:
        typer.echo(f"run_id={run_id!r} already completed — nothing to resume.")
        raise typer.Exit(code=1)

    orchestrator = build_orchestrator(settings, trace.config)

    typer.echo(f"Resuming debate {run_id!r} from round {len(trace.rounds) + 1}...")
    resumed_trace = asyncio.run(orchestrator.run(trace.request, run_id=run_id, resume=True))

    _print_summary(resumed_trace)


@app.command(name="list-runs")
def list_runs() -> None:
    """List all previously saved debate run_ids."""
    settings = get_settings()
    json_store = JsonStore(trace_json_dir=settings.trace_json_dir)

    run_ids = asyncio.run(json_store.list_runs())
    if not run_ids:
        typer.echo(f"No saved traces found in {settings.trace_json_dir}")
        return

    for run_id in run_ids:
        typer.echo(run_id)


if __name__ == "__main__":
    app()
