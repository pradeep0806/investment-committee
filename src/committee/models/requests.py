from typing import Literal

from pydantic import BaseModel, Field


class ThesisRequest(BaseModel):
    thesis: str = Field(min_length=1)
    entity: str | None = None
    priors: dict[str, str] | None = None


class DebateConfig(BaseModel):
    total_token_budget: int = Field(default=50_000, gt=0)
    num_rounds: int = Field(default=3, ge=2, le=3)
    agent_roles: list[str] | None = None
    # None (default) means "core 4 + all active custom personas" (Phase 1-B
    # decision — see orchestrator_factory.resolve_agents). Passing an
    # explicit list selects exactly those persona ids (built-in or custom)
    # and skips the "all active custom" default expansion, so a caller can
    # run a debate with only a subset. Distinct from agent_roles, which only
    # ever names built-in registry keys and predates persona-as-data — kept
    # unchanged so existing callers/tests aren't affected.
    agent_ids: list[str] | None = None
    conflict_resolution_strategy: Literal[
        "flag_unresolved", "confidence_weighted", "tie_breaker"
    ] = "flag_unresolved"
    # None means "use the server's Settings.convergence_*_threshold default"
    # (0.4/0.75) — same optional-override pattern as the llm_* fields below,
    # so a client (CLI/API/frontend) can try a different threshold for one
    # debate without a server restart. Was previously a non-optional float
    # with its own hardcoded default, which silently ignored
    # Settings.convergence_*_threshold entirely and had no way to represent
    # "use the server default" over the wire (sending null 422'd).
    convergence_low_threshold: float | None = Field(default=None, ge=0, le=1)
    convergence_high_threshold: float | None = Field(default=None, ge=0, le=1)

    # None means "use the server's Settings.min_viable_allocation_per_agent
    # default" — same optional-override pattern as the thresholds above.
    # What "viable" actually requires varies by provider/model: a real bug
    # found via a live debate against a weak local model (qwen3.5:9b) needed
    # roughly 2000+ tokens/agent to reliably produce valid structured
    # output, while a stronger hosted model succeeds well under 1000 for the
    # same schema — one hardcoded constant can't be right for every
    # provider, so this is a per-debate knob (raise it when running against
    # a weaker/local model) rather than baked into BudgetManager itself.
    min_viable_allocation_per_agent: int | None = Field(default=None, ge=1)

    # Per-debate LLM overrides — all optional, all None by default, meaning
    # "use whatever's configured server-side in .env/Settings." Letting a
    # caller (the CLI, the API, the frontend) override these per-request is
    # what makes trying a different provider/model/temperature a form field
    # rather than a server restart. API keys, Vertex project/location, and
    # timeouts stay server-only config — not something a client request
    # should be able to set.
    llm_provider: Literal["anthropic", "openai", "litellm"] | None = None
    llm_model: str | None = None
    llm_temperature: float | None = Field(default=None, ge=0, le=2)
    # Only meaningful for Gemini via the litellm provider — ignored by every
    # other provider/model. None means "let LiteLLMRawCaller derive a
    # default from max_tokens," matching the pre-existing behavior.
    llm_thinking_budget: int | None = Field(default=None, ge=0)
