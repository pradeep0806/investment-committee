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
    rebuttals: list[Rebuttal] | None = None
    tokens_used: int = Field(ge=0)
