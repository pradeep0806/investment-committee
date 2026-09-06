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
    conflict_resolution_strategy: Literal[
        "flag_unresolved", "confidence_weighted", "tie_breaker"
    ] = "flag_unresolved"
    convergence_low_threshold: float = Field(default=0.4, ge=0, le=1)
    convergence_high_threshold: float = Field(default=0.75, ge=0, le=1)

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
