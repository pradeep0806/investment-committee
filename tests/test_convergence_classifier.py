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


class TestAdversarialEchoVsGenuinePersuasion:
    """Interview follow-up (depth over breadth): prove the classifier holds
    against adversarial cases, not just plausible ones. Two synthetic agent
    personas reaching the *identical* conclusion (same stance, same round,
    against the same prior argument) — one must be flagged as an echo, the
    other must not, and both are demonstrated side by side so neither
    assertion is trivially true by construction alone."""

    _PRIOR_ARGUMENT = _output(
        "fundamentals",
        Stance.BUY,
        [
            "Q3 revenue grew 22% YoY, beating guidance by 4 points",
            "gross margin expanded 3pts on mix shift toward enterprise",
        ],
        round=1,
    )

    def _lazy_echo_agent(self) -> AgentOutput:
        """Restates fundamentals' conclusion citing the *same* evidence,
        only trivially reworded — different casing and surrounding
        whitespace, the exact normalization classify_round's `_normalize`
        already collapses (see convergence_classifier.py) — no new fact, no
        independent check. This is exactly the failure mode classify_round
        exists to catch: looks like a second, independent Buy vote, but the
        underlying evidence is identical to what's already on the table.

        Note this is a deliberately narrow adversarial case: the classifier
        compares *normalized literal text*, not meaning, so a lazy echo that
        paraphrases more heavily (different words, same fact) currently
        slips through as GENUINE — semantic echo detection via embedding
        similarity is the Stretch item that would close that gap. This
        fixture proves the literal-overlap case the classifier is actually
        built to catch, not a case it was never designed to catch."""
        return _output(
            "market_sentiment",
            Stance.BUY,
            [
                "  Q3 REVENUE grew 22% YoY, beating guidance by 4 points  ",
                "GROSS MARGIN expanded 3pts on mix shift toward enterprise",
            ],
            round=2,
        )

    def _genuinely_persuaded_agent(self) -> AgentOutput:
        """Reaches the same Buy conclusion as fundamentals, but the evidence
        is independently reasoned and non-overlapping — a different, later
        data point that happens to support the same stance, not a
        restatement of what fundamentals already said. This is the case a
        classifier that's too aggressive against agreement would wrongly
        punish as if it were an echo."""
        return _output(
            "macro_context",
            Stance.BUY,
            [
                "sector-wide enterprise IT spend forecast raised 2pts for next fiscal year",
                "two direct competitors reported decelerating growth this quarter, a relative tailwind",
            ],
            round=2,
        )

    def test_lazy_echo_agent_is_flagged_as_echo(self):
        result = classify_round(
            [self._lazy_echo_agent()], all_prior_outputs=[self._PRIOR_ARGUMENT]
        )
        assert result["market_sentiment"] == ConvergenceType.ECHO

    def test_genuinely_persuaded_agent_is_not_flagged_as_echo(self):
        result = classify_round(
            [self._genuinely_persuaded_agent()], all_prior_outputs=[self._PRIOR_ARGUMENT]
        )
        assert result["macro_context"] == ConvergenceType.GENUINE

    def test_both_cases_demonstrated_together_in_the_same_round(self):
        """The pair matters more than either case alone: same round, same
        target conclusion, same prior argument to compare against — proving
        the classifier discriminates between them rather than having a
        blanket bias toward flagging (or not flagging) agreement at all."""
        current_round = [self._lazy_echo_agent(), self._genuinely_persuaded_agent()]
        result = classify_round(current_round, all_prior_outputs=[self._PRIOR_ARGUMENT])

        assert result["market_sentiment"] == ConvergenceType.ECHO
        assert result["macro_context"] == ConvergenceType.GENUINE
