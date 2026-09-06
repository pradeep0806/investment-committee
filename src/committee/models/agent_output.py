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
    top_risk: str
    rebuttals: list[Rebuttal] | None = None
    tokens_used: int = Field(ge=0)
