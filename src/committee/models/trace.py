from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from committee.models.agent_output import AgentOutput
from committee.models.requests import DebateConfig, ThesisRequest
from committee.models.synthesis import ConvergenceSignal, DisagreementRecord, SynthesisMemo


class BudgetLedgerEntry(BaseModel):
    round: int = Field(ge=1)
    agent_id: str
    tokens_allocated: int = Field(ge=0)
    tokens_used: int = Field(ge=0)
    mode: Literal["explore", "balanced", "exploit"]
    # Which provider actually served this specific call — the primary, or
    # the fallback if the primary's retries were exhausted on a transient
    # error (llm/client.py). Default "" so ledger entries from before this
    # field existed (replayed on resume) still validate.
    provider_used: str = ""
    # True when this entry records a turn whose output was excluded from
    # the round (structured-output validation exhausted all retries) —
    # tokens_used still reflects real spend across the failed attempts
    # (see LLMValidationError.total_tokens_used), it just never produced a
    # usable AgentOutput. Default False so ledger entries from before this
    # field existed still validate.
    excluded: bool = False


class RoundRecord(BaseModel):
    round: int = Field(ge=1)
    agent_outputs: list[AgentOutput]
    convergence_signal: ConvergenceSignal
    disagreements: list[DisagreementRecord] = Field(default_factory=list)
    ledger_entries: list[BudgetLedgerEntry]
    started_at: datetime
    ended_at: datetime


class DebateTrace(BaseModel):
    run_id: str
    request: ThesisRequest
    config: DebateConfig
    rounds: list[RoundRecord] = Field(default_factory=list)
    all_agent_outputs: list[AgentOutput] = Field(default_factory=list)
    budget_ledger: list[BudgetLedgerEntry] = Field(default_factory=list)
    convergence_signals: list[ConvergenceSignal] = Field(default_factory=list)
    disagreements: list[DisagreementRecord] = Field(default_factory=list)
    synthesis: SynthesisMemo | None = None
    started_at: datetime
    ended_at: datetime | None = None
    total_tokens_used: int = 0
