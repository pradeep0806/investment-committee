"""Prometheus metric registry.

Metrics are added here at the exact point in the build where the orchestrator
component that produces them is written (CLAUDE.md §10) — this module grows
across steps 3-9, never as one bolted-on later pass. Only `debate_duration_seconds`
exists as of step 3; later steps append their own metrics alongside it.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()

debate_duration_seconds = Histogram(
    "debate_duration_seconds",
    "Wall-clock duration of a full debate run, from orchestrator start to synthesis.",
    registry=REGISTRY,
)

debate_tokens_used_total = Counter(
    "debate_tokens_used_total",
    "Actual tokens consumed, the raw material for a budget-by-agent panel.",
    ["agent", "round"],
    registry=REGISTRY,
)

disagreements_detected_total = Counter(
    "disagreements_detected_total",
    "Disagreements detected per debate run, incremented every round they're found.",
    ["run_id"],
    registry=REGISTRY,
)

debate_convergence_score = Gauge(
    "debate_convergence_score",
    "Composite convergence score for a run, updated after every round. The "
    "single number the explore-exploit controller decides mode on.",
    ["run_id"],
    registry=REGISTRY,
)

mode_transitions_total = Counter(
    "mode_transitions_total",
    "Explore/balanced/exploit mode transitions, labeled by from/to mode.",
    ["from_mode", "to_mode"],
    registry=REGISTRY,
)

debate_active_agent = Gauge(
    "debate_active_agent",
    "1 while an agent is reasoning, else 0 — answers 'which agent is reasoning "
    "right now' for a live dashboard panel. Sourced from the same event hook "
    "the SSE stream consumes (agent_reasoning_start / agent_reasoning_end).",
    ["run_id", "agent"],
    registry=REGISTRY,
)

llm_call_errors_total = Counter(
    "llm_call_errors_total",
    "LLM API call failures, by provider.",
    ["provider"],
    registry=REGISTRY,
)

llm_call_latency_seconds = Histogram(
    "llm_call_latency_seconds",
    "Latency of a single LLM API call, by provider.",
    ["provider"],
    registry=REGISTRY,
)

budget_overrun_tokens_total = Counter(
    "budget_overrun_tokens_total",
    "Tokens used beyond what was allocated, by agent — the single most "
    "diagnostic signal for whether token_budget is actually behaving as an "
    "advisory number rather than a cap the model respects. Zero increments "
    "means every agent stayed within its allocation.",
    ["agent"],
    registry=REGISTRY,
)
