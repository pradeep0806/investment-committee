from enum import Enum

from pydantic import BaseModel, Field


class Stance(str, Enum):
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"
    PASS = "Pass"


class Rebuttal(BaseModel):
    target_agent_id: str
    round: int = Field(ge=1)
    rebuttal_text: str


class AgentOutput(BaseModel):
    agent_id: str
    # Human-readable label for provenance display (e.g. a custom persona's
    # name, since agent_id for a dynamic agent is an opaque uuid hex, not a
    # readable slug like the built-ins' "fundamentals"). Optional/default-None
    # so existing fixtures and any agent that doesn't set it still validate;
    # the orchestrator/trace reader can always fall back to agent_id alone.
    agent_name: str | None = None
    round: int = Field(ge=1)
    stance: Stance
    confidence: int = Field(ge=0, le=100)
    key_factors: list[str]
    # Concrete cited facts/data points backing this round's argument —
    # distinct from key_factors (which are short tag labels used for Jaccard
    # overlap). Convergence classification needs something more specific than
    # tag overlap to tell "independently found the same evidence" apart from
    # "restated another agent's evidence with no new support" (see
    # orchestration/convergence_classifier.py). Optional/default-empty so
    # existing fixtures and any agent that doesn't populate it still validate.
    evidence: list[str] = Field(default_factory=list)
    top_risk: str
    # Plain-language digest of why this agent landed on its stance — derived
    # from the same structured-output call as everything else above, not a
    # second LLM round trip. Distinct from (and much shorter than) the full
    # reasoning trace made up of key_factors/evidence/top_risk. Bounded so it
    # stays a digest rather than a restatement of that trace.
    executive_summary: str = Field(default="", max_length=280)
    rebuttals: list[Rebuttal] | None = None
    tokens_used: int = Field(ge=0)
