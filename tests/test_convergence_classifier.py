from committee.models.agent_output import AgentOutput, Stance
from committee.models.synthesis import ConvergenceType
from committee.orchestration.convergence_classifier import classify_round


def _output(agent_id: str, stance: Stance, evidence: list[str], round: int = 1) -> AgentOutput:
    return AgentOutput(
        agent_id=agent_id,
        round=round,
        stance=stance,
        confidence=70,
        key_factors=["growth"],
        evidence=evidence,
        top_risk="some risk",
        tokens_used=500,
    )


class TestClassifyRound:
    def test_no_prior_outputs_yields_none_for_everyone(self):
        current = [_output("a", Stance.BUY, ["Q3 revenue up 22%"])]
        result = classify_round(current, all_prior_outputs=[])
        assert result == {"a": ConvergenceType.NONE}

    def test_new_stance_with_no_matching_prior_stance_is_none(self):
        prior = [_output("a", Stance.SELL, ["churn up 4pts"], round=1)]
        current = [_output("b", Stance.BUY, ["Q3 revenue up 22%"], round=2)]
        result = classify_round(current, all_prior_outputs=prior)
        assert result["b"] == ConvergenceType.NONE

    def test_same_conclusion_with_only_previously_stated_evidence_is_echo(self):
        prior = [_output("a", Stance.BUY, ["Q3 revenue up 22%"], round=1)]
        current = [_output("b", Stance.BUY, ["Q3 revenue up 22%"], round=2)]
        result = classify_round(current, all_prior_outputs=prior)
        assert result["b"] == ConvergenceType.ECHO

    def test_same_conclusion_with_new_evidence_is_genuine(self):
        prior = [_output("a", Stance.BUY, ["Q3 revenue up 22%"], round=1)]
        current = [_output("b", Stance.BUY, ["Q3 revenue up 22%", "margin expanded 3pts"], round=2)]
        result = classify_round(current, all_prior_outputs=prior)
        assert result["b"] == ConvergenceType.GENUINE

    def test_echo_of_own_prior_round_is_still_echo(self):
        prior = [_output("a", Stance.BUY, ["Q3 revenue up 22%"], round=1)]
        current = [_output("a", Stance.BUY, ["Q3 revenue up 22%"], round=2)]
        result = classify_round(current, all_prior_outputs=prior)
        assert result["a"] == ConvergenceType.ECHO

    def test_evidence_comparison_is_normalized_case_and_whitespace(self):
        prior = [_output("a", Stance.BUY, ["Q3 revenue up 22%"], round=1)]
        current = [_output("b", Stance.BUY, [" q3 REVENUE up 22% "], round=2)]
        result = classify_round(current, all_prior_outputs=prior)
        assert result["b"] == ConvergenceType.ECHO

    def test_evidence_matched_against_same_stance_only_not_opposing_stance(self):
        """Sell-side evidence already on the table doesn't make a fresh Buy
        claim citing the same text an echo — it's only an echo relative to
        arguments that reached the *same* conclusion."""
        prior = [_output("a", Stance.SELL, ["margins compressing"], round=1)]
        current = [_output("b", Stance.BUY, ["margins compressing"], round=2)]
        result = classify_round(current, all_prior_outputs=prior)
        assert result["b"] == ConvergenceType.NONE

    def test_reaches_two_rounds_back_not_just_immediately_prior_round(self):
        round1 = [_output("a", Stance.BUY, ["Q3 revenue up 22%"], round=1)]
        round2 = [_output("b", Stance.SELL, ["churn up"], round=2)]
        current = [_output("c", Stance.BUY, ["Q3 revenue up 22%"], round=3)]
        result = classify_round(current, all_prior_outputs=round1 + round2)
        assert result["c"] == ConvergenceType.ECHO
