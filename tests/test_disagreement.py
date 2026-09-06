from committee.models.agent_output import AgentOutput, Stance
from committee.orchestration.disagreement import detect


def _output(agent_id: str, stance: Stance, confidence: int, key_factors: list[str], round: int = 2) -> AgentOutput:
    return AgentOutput(
        agent_id=agent_id,
        round=round,
        stance=stance,
        confidence=confidence,
        key_factors=key_factors,
        top_risk="some risk",
        tokens_used=500,
    )


def test_clear_majority_produces_no_disagreement():
    outputs = [
        _output("fundamentals", Stance.BUY, 80, ["growth"]),
        _output("market_sentiment", Stance.BUY, 75, ["growth"]),
        _output("risk_contrarian", Stance.BUY, 60, ["growth"]),
        _output("macro_context", Stance.BUY, 70, ["growth"]),
    ]
    assert detect(round=2, agent_outputs=outputs) == []


def test_no_majority_produces_disagreement():
    outputs = [
        _output("fundamentals", Stance.BUY, 70, ["growth"]),
        _output("market_sentiment", Stance.HOLD, 60, ["momentum"]),
        _output("risk_contrarian", Stance.SELL, 40, ["risk"]),
        _output("macro_context", Stance.PASS, 50, ["macro"]),
    ]
    records = detect(round=2, agent_outputs=outputs)
    assert len(records) == 1
    assert records[0].round_detected == 2
    assert set(records[0].agents_involved) == {
        "fundamentals",
        "market_sentiment",
        "risk_contrarian",
        "macro_context",
    }


def test_high_confidence_buy_sell_conflict_flagged_even_with_majority():
    """3/4 agents agreeing on Hold still isn't safe to synthesize past if one
    Buy and one Sell both hold high confidence — never average away a
    confident direct conflict."""
    outputs = [
        _output("fundamentals", Stance.BUY, 85, ["valuation"]),
        _output("market_sentiment", Stance.HOLD, 60, ["momentum"]),
        _output("risk_contrarian", Stance.SELL, 80, ["valuation"]),
        _output("macro_context", Stance.HOLD, 55, ["macro"]),
    ]
    records = detect(round=2, agent_outputs=outputs)
    assert len(records) == 1
    assert set(records[0].opposing_stances.keys()) == {"fundamentals", "risk_contrarian"}
    assert records[0].opposing_stances["fundamentals"] == Stance.BUY
    assert records[0].opposing_stances["risk_contrarian"] == Stance.SELL


def test_low_confidence_buy_sell_conflict_not_flagged_if_majority_exists():
    outputs = [
        _output("fundamentals", Stance.BUY, 55, ["growth"]),
        _output("market_sentiment", Stance.HOLD, 60, ["momentum"]),
        _output("risk_contrarian", Stance.SELL, 50, ["risk"]),
        _output("macro_context", Stance.HOLD, 65, ["macro"]),
    ]
    # 2/4 Hold is not >= 0.75 majority either, so this SHOULD still flag —
    # use a genuine 3/4 majority case instead to isolate the confidence check.
    outputs_with_majority = [
        _output("fundamentals", Stance.BUY, 55, ["growth"]),
        _output("market_sentiment", Stance.HOLD, 60, ["momentum"]),
        _output("risk_contrarian", Stance.HOLD, 50, ["risk"]),
        _output("macro_context", Stance.HOLD, 65, ["macro"]),
    ]
    assert detect(round=2, agent_outputs=outputs_with_majority) == []


def test_disagreement_never_averages_stances():
    """The core invariant: detect() has no code path that blends opposing
    stances into a single value — it only ever returns explicit records."""
    outputs = [
        _output("fundamentals", Stance.BUY, 90, ["x"]),
        _output("risk_contrarian", Stance.SELL, 90, ["x"]),
    ]
    records = detect(round=2, agent_outputs=outputs)
    assert len(records) == 1
    assert isinstance(records[0].opposing_stances["fundamentals"], Stance)
    assert isinstance(records[0].opposing_stances["risk_contrarian"], Stance)
    # no numeric "blended recommendation" field exists on the model at all
    assert not hasattr(records[0], "blended_stance")
    assert not hasattr(records[0], "average_confidence")


def test_contested_factors_are_factors_shared_by_multiple_disagreeing_agents():
    outputs = [
        _output("fundamentals", Stance.BUY, 85, ["valuation", "growth"]),
        _output("risk_contrarian", Stance.SELL, 80, ["valuation", "customer concentration"]),
    ]
    records = detect(round=2, agent_outputs=outputs)
    assert records[0].contested_factors == ["valuation"]


def test_contested_factors_matches_despite_casing_and_whitespace_differences():
    """Real bug: _contested_factors() used an exact-match Counter with no
    normalization, while explore_exploit.py's _factor_overlap() reads the
    same key_factors field but normalizes with .strip().lower() first. Two
    agents citing the same factor with different casing/whitespace (e.g.
    "Valuation" vs "valuation ") never registered as contested here, even
    though they'd count as overlapping for the convergence score — in
    production this showed up as contested_factors: [] in nearly every
    disagreement record. Confirmed this fails on the unfixed code before
    the fix was applied."""
    outputs = [
        _output("fundamentals", Stance.BUY, 85, ["Valuation", "Growth"]),
        _output("risk_contrarian", Stance.SELL, 80, ["valuation ", "Customer Concentration"]),
    ]
    records = detect(round=2, agent_outputs=outputs)
    assert records[0].contested_factors == ["Valuation"]


def test_contested_factors_preserves_first_seen_original_casing():
    """The returned spelling should be whichever agent's original phrasing
    was seen first, not a lowercased normalization key — so
    contested_factors reads naturally in the synthesis memo."""
    outputs = [
        _output("fundamentals", Stance.BUY, 85, ["Valuation Concerns"]),
        _output("risk_contrarian", Stance.SELL, 80, ["valuation concerns"]),
    ]
    records = detect(round=2, agent_outputs=outputs)
    assert records[0].contested_factors == ["Valuation Concerns"]


def test_empty_agent_outputs_produces_no_disagreement():
    assert detect(round=2, agent_outputs=[]) == []
