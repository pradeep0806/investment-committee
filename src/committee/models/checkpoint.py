from datetime import datetime

from pydantic import BaseModel, Field


class DebateCheckpoint(BaseModel):
    """Written to MongoDB after every round completes (CLAUDE.md §1.3's
    incremental-persistence requirement, extended per the hardening task):
    enough state for a crashed process to resume a debate from the last
    completed round rather than restarting it from scratch. Distinct from
    DebateTrace — the trace is the full structured record (all outputs,
    signals, disagreements); the checkpoint is the minimal pointer a resume
    path actually needs to know where to pick back up."""

    run_id: str
    last_completed_round: int = Field(ge=0)
    phase: str  # "in_progress" | "complete"
    budget_remaining: int
    updated_at: datetime
