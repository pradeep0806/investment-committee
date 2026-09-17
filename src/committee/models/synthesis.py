from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from committee.models.agent_output import Stance


class ConvergenceType(str, Enum):
    """Per-agent classification of *why* this round's output agrees with a
    prior one — see orchestration/convergence_classifier.py. Only GENUINE
    convergence should count toward the explore->exploit trigger; an ECHO
    (same conclusion, no evidence beyond what was already on the table) is
    not trustworthy convergence, even though it looks identical to genuine
    convergence under stance/factor-tag comparison alone."""

    GENUINE = "genuine"
    ECHO = "echo"
    NONE = "none"


class ConvergenceSignal(BaseModel):
    round: int = Field(ge=1)
    stance_agreement: float = Field(ge=0, le=1)
    factor_overlap: float = Field(ge=0, le=1)
    confidence_spread: float = Field(ge=0, le=1)
    composite_score: float = Field(ge=0, le=1)
    mode_selected: Literal["explore", "balanced", "exploit"]
    # agent_id -> ConvergenceType for this round, additive field. Empty dict
    # for round 1 (nothing prior to compare against) or any caller that
    # doesn't pass classification into score().
    convergence_types: dict[str, ConvergenceType] = Field(default_factory=dict)


class DisagreementRecord(BaseModel):
    round_detected: int = Field(ge=1)
    agents_involved: list[str]
    opposing_stances: dict[str, Stance]
    contested_factors: list[str]
    resolution_strategy_applied: str | None = None
    resolved: bool = False


class DissentEntry(BaseModel):
    """One agent whose final-round stance differs from the committee's final
    `recommendation` — named and reasoned individually, never blended into
    the majority conclusion (CLAUDE.md §5 step 6)."""

    agent_id: str
    agent_name: str | None = None
    stance: Stance
    # Reused from the agent's own executive_summary when it set one,
    # otherwise its top_risk — never a newly generated string, since this is
    # pure aggregation over data the agent already produced.
    reason: str


class SynthesisMemo(BaseModel):
    recommendation: Stance
    confidence: int = Field(ge=0, le=100)
    supporting_agents: list[str]
    dissenting_agents: list[str]
    dissent_appendix: str | None = None
    reasoning_trace_refs: list[str]
    # agent_id -> executive_summary for every final-round agent, an "at a
    # glance" companion to supporting_agents/dissenting_agents so a reader
    # doesn't have to open the full trace to see why each agent landed where
    # it did. Additive/default-empty so existing fixtures still validate.
    agent_summaries: dict[str, str] = Field(default_factory=dict)
    # Structured, named view of every final-round agent whose stance differs
    # from `recommendation` — empty when every agent converged, in which case
    # dissenting_view_note explicitly says so rather than leaving the absence
    # of dissent implicit. Distinct from dissenting_agents (bare id list) and
    # dissent_appendix (strategy-authored free-text paragraph): this is the
    # named, per-agent, non-blended record the Core disagreement-handling
    # requirement asks for.
    dissenting_view: list[DissentEntry] = Field(default_factory=list)
    dissenting_view_note: str = ""
    # agent_ids whose final-round argument was classified ECHO (see
    # orchestration/convergence_classifier.py) and were therefore excluded
    # from the majority-stance vote, supporting_agents, and the average
    # confidence on the clean-consensus path — down-weighted, not silently
    # averaged in as if their vote counted the same as a genuine one. Kept
    # here (rather than just dropped) so an echoed agent's presence in the
    # round is still visible in the memo, not erased. Always empty on the
    # disagreement/conflict-resolution path (out of scope for this task —
    # see CLAUDE.md's interview-follow-up brief) and for any caller that
    # doesn't pass convergence_types into synthesize().
    echoed_agents: list[str] = Field(default_factory=list)
