from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from committee.models import (
    AgentOutput,
    BudgetLedgerEntry,
    ConvergenceSignal,
    DebateConfig,
    DebateTrace,
    DisagreementRecord,
    RoundRecord,
    Stance,
    SynthesisMemo,
    ThesisRequest,
)


def _sample_agent_output(**overrides) -> AgentOutput:
    fields = dict(
        agent_id="fundamentals",
        round=1,
        stance=Stance.BUY,
        confidence=80,
        key_factors=["revenue growth", "margin expansion"],
        top_risk="customer concentration",
        tokens_used=1200,
    )
    fields.update(overrides)
    return AgentOutput(**fields)


def test_thesis_request_round_trip():
    req = ThesisRequest(thesis="NovaTech is undervalued", entity="NovaTech Inc.")
    restored = ThesisRequest.model_validate_json(req.model_dump_json())
    assert restored == req


def test_thesis_request_requires_nonempty_thesis():
    with pytest.raises(ValidationError):
        ThesisRequest(thesis="")


def test_debate_config_defaults():
    config = DebateConfig()
    assert config.total_token_budget == 50_000
    assert config.num_rounds == 3
    assert config.conflict_resolution_strategy == "flag_unresolved"


@pytest.mark.parametrize("num_rounds", [1, 4])
def test_debate_config_rejects_out_of_range_rounds(num_rounds):
    with pytest.raises(ValidationError):
        DebateConfig(num_rounds=num_rounds)


def test_debate_config_rejects_non_positive_budget():
    with pytest.raises(ValidationError):
        DebateConfig(total_token_budget=0)


def test_stance_enum_values():
    assert {s.value for s in Stance} == {"Buy", "Hold", "Sell", "Pass"}


def test_agent_output_round_trip():
    output = _sample_agent_output()
    restored = AgentOutput.model_validate_json(output.model_dump_json())
    assert restored == output


@pytest.mark.parametrize("confidence", [-1, 101])
def test_agent_output_rejects_out_of_range_confidence(confidence):
    with pytest.raises(ValidationError):
        _sample_agent_output(confidence=confidence)


def test_agent_output_rejects_executive_summary_over_max_length():
    with pytest.raises(ValidationError):
        _sample_agent_output(executive_summary="x" * 281)


def test_agent_output_accepts_executive_summary_within_bound():
    output = _sample_agent_output(executive_summary="Strong growth outweighs concentration risk.")
    assert output.executive_summary == "Strong growth outweighs concentration risk."


def test_agent_output_rejects_invalid_stance():
    with pytest.raises(ValidationError):
        AgentOutput(
            agent_id="fundamentals",
            round=1,
            stance="Strong Buy",
            confidence=80,
            key_factors=["x"],
            top_risk="y",
            tokens_used=100,
        )


def test_budget_ledger_entry_round_trip():
    entry = BudgetLedgerEntry(
        round=1, agent_id="fundamentals", tokens_allocated=1000, tokens_used=950, mode="explore"
    )
    restored = BudgetLedgerEntry.model_validate_json(entry.model_dump_json())
    assert restored == entry


def test_convergence_signal_rejects_out_of_range_score():
    with pytest.raises(ValidationError):
        ConvergenceSignal(
            round=1,
            stance_agreement=0.5,
            factor_overlap=0.5,
            confidence_spread=0.5,
            composite_score=1.5,
            mode_selected="balanced",
        )


def test_disagreement_record_round_trip():
    record = DisagreementRecord(
        round_detected=2,
        agents_involved=["fundamentals", "risk_contrarian"],
        opposing_stances={"fundamentals": Stance.BUY, "risk_contrarian": Stance.SELL},
        contested_factors=["valuation multiple"],
    )
    restored = DisagreementRecord.model_validate_json(record.model_dump_json())
    assert restored == record
    assert restored.resolved is False


def test_synthesis_memo_dissent_appendix_optional():
    memo = SynthesisMemo(
        recommendation=Stance.BUY,
        confidence=70,
        supporting_agents=["fundamentals", "market_sentiment"],
        dissenting_agents=[],
        reasoning_trace_refs=["round2:fundamentals"],
    )
    assert memo.dissent_appendix is None


def test_debate_trace_round_trip():
    now = datetime.now(timezone.utc)
    output = _sample_agent_output()
    signal = ConvergenceSignal(
        round=1,
        stance_agreement=1.0,
        factor_overlap=0.5,
        confidence_spread=0.1,
        composite_score=0.83,
        mode_selected="exploit",
    )
    ledger_entry = BudgetLedgerEntry(
        round=1, agent_id="fundamentals", tokens_allocated=1000, tokens_used=1200, mode="exploit"
    )
    round_record = RoundRecord(
        round=1,
        agent_outputs=[output],
        convergence_signal=signal,
        ledger_entries=[ledger_entry],
        started_at=now,
        ended_at=now,
    )
    trace = DebateTrace(
        run_id="run-1",
        request=ThesisRequest(thesis="Test thesis"),
        config=DebateConfig(),
        rounds=[round_record],
        all_agent_outputs=[output],
        budget_ledger=[ledger_entry],
        convergence_signals=[signal],
        started_at=now,
        total_tokens_used=1200,
    )
    restored = DebateTrace.model_validate_json(trace.model_dump_json())
    assert restored == trace
    assert restored.synthesis is None
    assert restored.disagreements == []
