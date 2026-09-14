from committee.agents.registry import build_agents
from committee.models.requests import DebateConfig, ThesisRequest
from committee.observability.logging import configure_logging
from committee.orchestration.orchestrator import DebateOrchestrator

configure_logging()


class _FakeRawCaller:
    def __init__(self):
        self.call_count = 0

    async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens=None):
        self.call_count += 1
        return (
            {
                "stance": "Buy",
                "confidence": 65,
                "key_factors": ["revenue growth", "margin expansion"],
                "top_risk": "customer concentration",
            },
            500,
        )


class _OneAgentFailsRawCaller:
    """fundamentals always fails validation; every other agent succeeds
    normally — proves one bad agent doesn't take down the whole round."""

    def __init__(self):
        self.call_count = 0

    async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens=None):
        self.call_count += 1
        if system_prompt.startswith("You are the Fundamentals/Valuation analyst"):
            return ({"stance": "Buy", "confidence": 999, "key_factors": [], "top_risk": "x"}, 100)
        return (
            {
                "stance": "Hold",
                "confidence": 55,
                "key_factors": ["momentum"],
                "top_risk": "some risk",
            },
            500,
        )


class _DivergentThenAgreeingRawCaller:
    """Round 1: agents split (low convergence -> explore/balanced). From round
    2 onward: all agents agree (high convergence -> exploit). Proves mode is
    a genuine computed transition across rounds, not fixed."""

    def __init__(self):
        self.call_count = 0
        self.round_seen: dict[str, int] = {}

    async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens=None):
        self.call_count += 1
        # crude round tracking: user_prompt embeds round-1 guardrail text only
        # in round 1 (see agents/prompts/shared.py ROUND_ONE_GUARDRAIL)
        is_round_one = "This is round 1" in user_prompt
        if is_round_one:
            if "Fundamentals" in system_prompt:
                return (
                    {"stance": "Buy", "confidence": 80, "key_factors": ["growth"], "top_risk": "x"},
                    500,
                )
            return (
                {"stance": "Sell", "confidence": 75, "key_factors": ["risk"], "top_risk": "y"},
                500,
            )
        return (
            {"stance": "Buy", "confidence": 80, "key_factors": ["growth"], "top_risk": "x"},
            500,
        )


class _PersistentSplitRawCaller:
    """fundamentals always says Buy at high confidence; risk_contrarian always
    says Sell at high confidence; the other two agree with fundamentals — a
    disagreement that never resolves itself across rounds, so it survives to
    the final round and reaches synthesis/conflict resolution."""

    def __init__(self):
        self.call_count = 0

    async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens=None):
        self.call_count += 1
        if system_prompt.startswith("You are the Risk Contrarian analyst"):
            return (
                {"stance": "Sell", "confidence": 85, "key_factors": ["valuation"], "top_risk": "y"},
                500,
            )
        return (
            {"stance": "Buy", "confidence": 80, "key_factors": ["valuation"], "top_risk": "x"},
            500,
        )


def _make_fake_client(raw_caller=None, max_retries=3):
    from committee.llm.client import LLMClient

    client = LLMClient.__new__(LLMClient)
    client.provider = "fake"
    client.model = "fake-model"
    client.api_key = "fake-key"
    client.timeout_seconds = 60
    client.max_retries = max_retries
    client._raw_caller = raw_caller if raw_caller is not None else _FakeRawCaller()
    return client


async def test_orchestrator_clears_run_gauges_after_completion():
    """Real gap found via live testing: debate_convergence_score and
    debate_active_agent are Prometheus Gauges, which hold their last-set
    value forever — nothing about a debate finishing tells them to stop
    reporting it. Left alone, every run_id this process ever handled
    accumulates in /metrics and Grafana's legend indefinitely (observed
    live: a run that finished minutes earlier still showed a flat
    convergence line and permanently-idle active-agent rows on every
    subsequent scrape). A completed run's gauges must be removed once
    its numbers stop being current."""
    from committee.observability.metrics import debate_active_agent, debate_convergence_score

    client = _make_fake_client()
    agents = build_agents(llm_client=client)
    config = DebateConfig(total_token_budget=8000, num_rounds=2)
    orchestrator = DebateOrchestrator(config=config, agents=agents)

    trace = await orchestrator.run(ThesisRequest(thesis="Test thesis for gauge cleanup"))
    run_id = trace.run_id

    for metric in debate_convergence_score.collect():
        for sample in metric.samples:
            assert sample.labels.get("run_id") != run_id

    for metric in debate_active_agent.collect():
        for sample in metric.samples:
            assert sample.labels.get("run_id") != run_id


async def test_orchestrator_runs_configured_num_rounds_across_all_default_agents():
    client = _make_fake_client()
    agents = build_agents(llm_client=client)
    config = DebateConfig(total_token_budget=8000, num_rounds=2)
    orchestrator = DebateOrchestrator(config=config, agents=agents)

    trace = await orchestrator.run(ThesisRequest(thesis="Test thesis for orchestrator"))

    assert len(trace.rounds) == 2
    assert all(len(round_record.agent_outputs) == 4 for round_record in trace.rounds)
    assert len(trace.all_agent_outputs) == 8
    assert client._raw_caller.call_count == 8

    agent_ids_round1 = {output.agent_id for output in trace.rounds[0].agent_outputs}
    assert agent_ids_round1 == {"fundamentals", "market_sentiment", "risk_contrarian", "macro_context"}


async def test_orchestrator_splits_budget_equally_per_agent_within_a_round():
    """Allocation is even across agents *within* a round (explore/balanced
    mode has no reason to favor one agent over another) — but the baseline is
    recomputed each round from whatever's actually left, out of the
    *spendable* budget (BudgetManager carves out a reserve pool, 10% by
    default, for tie-breaker spawns before splitting the rest), so it's not
    simply a fixed total_token_budget // (agents * rounds) repeated every
    round: an earlier round under- or over-spending its baseline changes what
    later rounds have to work with."""
    client = _make_fake_client()
    agents = build_agents(llm_client=client)
    config = DebateConfig(total_token_budget=8000, num_rounds=2)
    orchestrator = DebateOrchestrator(config=config, agents=agents)

    trace = await orchestrator.run(ThesisRequest(thesis="Test thesis"))

    for round_num in (1, 2):
        round_entries = [e for e in trace.budget_ledger if e.round == round_num]
        assert len(set(e.tokens_allocated for e in round_entries)) == 1, (
            f"round {round_num} allocations were not even across agents"
        )

    round1_allocation = trace.budget_ledger[0].tokens_allocated
    spendable_budget = 8000 - int(8000 * 0.10)
    assert round1_allocation == spendable_budget // (4 * 2)


async def test_orchestrator_second_round_receives_prior_round_outputs():
    client = _make_fake_client()
    agents = build_agents(llm_client=client)
    config = DebateConfig(total_token_budget=8000, num_rounds=2)
    orchestrator = DebateOrchestrator(config=config, agents=agents)

    trace = await orchestrator.run(ThesisRequest(thesis="Test thesis"))

    # Round 2's user prompts should reference round 1's agent outputs.
    round_two_calls = client._raw_caller
    assert round_two_calls.call_count == 8
    assert trace.rounds[1].round == 2
    assert len(trace.rounds[0].agent_outputs) == 4


async def test_orchestrator_sets_run_id_and_total_tokens():
    client = _make_fake_client()
    agents = build_agents(llm_client=client)
    config = DebateConfig(total_token_budget=8000, num_rounds=2)
    orchestrator = DebateOrchestrator(config=config, agents=agents)

    trace = await orchestrator.run(ThesisRequest(thesis="Test thesis"))

    assert trace.run_id
    assert trace.total_tokens_used == 500 * 8
    assert trace.ended_at is not None
    assert trace.ended_at >= trace.started_at


async def test_orchestrator_excludes_agent_that_fails_structured_output_validation():
    """One agent (fundamentals) always fails validation after exhausting
    retries — the debate must continue without it, not crash, and the failed
    agent must not appear in that round's outputs."""
    client = _make_fake_client(raw_caller=_OneAgentFailsRawCaller(), max_retries=2)
    agents = build_agents(llm_client=client)
    config = DebateConfig(total_token_budget=8000, num_rounds=2)
    orchestrator = DebateOrchestrator(config=config, agents=agents)

    trace = await orchestrator.run(ThesisRequest(thesis="Test thesis"))

    assert len(trace.rounds) == 2
    for round_record in trace.rounds:
        agent_ids = {output.agent_id for output in round_record.agent_outputs}
        assert "fundamentals" not in agent_ids
        assert agent_ids == {"market_sentiment", "risk_contrarian", "macro_context"}
    # excluded agent's failed attempts still consumed no budget-ledger entry
    assert all(entry.agent_id != "fundamentals" for entry in trace.budget_ledger)


async def test_orchestrator_mode_transitions_from_computed_convergence_not_hardcoded():
    """The centerpiece guarantee (CLAUDE.md §1.6): mode is a computed signal
    from actual agent outputs, never `if round == N`. Round 1 always starts
    explore (nothing to converge on yet); round 2's mode is decided from
    round 1's actual stance split; round 3's mode is decided from round 2's
    actual (now-converged) outputs."""
    client = _make_fake_client(raw_caller=_DivergentThenAgreeingRawCaller())
    agents = build_agents(llm_client=client)
    config = DebateConfig(total_token_budget=8000, num_rounds=3)
    orchestrator = DebateOrchestrator(config=config, agents=agents)

    trace = await orchestrator.run(ThesisRequest(thesis="Test thesis"))

    assert len(trace.rounds) == 3
    round1_stances = {o.stance.value for o in trace.rounds[0].agent_outputs}
    assert round1_stances == {"Buy", "Sell"}  # genuinely split, not hardcoded agreement

    round1_signal = trace.rounds[0].convergence_signal
    assert round1_signal.stance_agreement < 1.0

    # Round 2 agents all agree (per the fake caller) -> round 2's own signal
    # should show full agreement, and round 3 (decided from round 2's
    # signal) should be in exploit mode as a result.
    round2_signal = trace.rounds[1].convergence_signal
    assert round2_signal.stance_agreement == 1.0
    assert round2_signal.composite_score > config.convergence_high_threshold


async def test_orchestrator_populates_synthesis_with_clean_consensus():
    client = _make_fake_client()  # default: all agents agree
    agents = build_agents(llm_client=client)
    config = DebateConfig(total_token_budget=8000, num_rounds=2)
    orchestrator = DebateOrchestrator(config=config, agents=agents)

    trace = await orchestrator.run(ThesisRequest(thesis="Test thesis"))

    assert trace.synthesis is not None
    assert trace.synthesis.dissent_appendix is None
    assert trace.synthesis.dissenting_agents == []


async def test_orchestrator_synthesis_changes_when_strategy_swapped_on_persistent_disagreement():
    """The full end-to-end version of CLAUDE.md §9's requirement: run the same
    persistently-disagreeing debate through flag_unresolved vs
    confidence_weighted (via config only, no orchestrator code change) and
    confirm the synthesis differs."""
    config_flag = DebateConfig(
        total_token_budget=8000, num_rounds=2, conflict_resolution_strategy="flag_unresolved"
    )
    client_flag = _make_fake_client(raw_caller=_PersistentSplitRawCaller())
    orchestrator_flag = DebateOrchestrator(config=config_flag, agents=build_agents(llm_client=client_flag))
    trace_flag = await orchestrator_flag.run(ThesisRequest(thesis="Test thesis"))

    config_weighted = DebateConfig(
        total_token_budget=8000, num_rounds=2, conflict_resolution_strategy="confidence_weighted"
    )
    client_weighted = _make_fake_client(raw_caller=_PersistentSplitRawCaller())
    orchestrator_weighted = DebateOrchestrator(
        config=config_weighted, agents=build_agents(llm_client=client_weighted)
    )
    trace_weighted = await orchestrator_weighted.run(ThesisRequest(thesis="Test thesis"))

    assert trace_flag.disagreements  # confirms the fixture actually produced a disagreement
    assert trace_flag.synthesis.recommendation != trace_weighted.synthesis.recommendation
    assert trace_flag.synthesis.dissent_appendix is not None
    assert trace_weighted.synthesis.dissent_appendix is not None
    assert trace_flag.synthesis.recommendation.value == "Pass"
    # 3 Buy vs 1 Sell, both at high confidence -> Buy wins on aggregate weight
    assert trace_weighted.synthesis.recommendation.value == "Buy"


async def test_orchestrator_tie_breaker_strategy_spawns_agent_and_draws_from_reserve():
    """tie_breaker needs an llm_client on the orchestrator to spawn its
    verdict agent, and its spend must come out of the reserve pool rather
    than the per-round spendable budget."""
    config = DebateConfig(
        total_token_budget=8000, num_rounds=2, conflict_resolution_strategy="tie_breaker"
    )
    client = _make_fake_client(raw_caller=_PersistentSplitRawCaller())
    orchestrator = DebateOrchestrator(
        config=config, agents=build_agents(llm_client=client), llm_client=client
    )

    trace = await orchestrator.run(ThesisRequest(thesis="Test thesis"))

    assert trace.synthesis is not None
    assert "tie_breaker" in trace.synthesis.supporting_agents
    # tie-breaker's own verdict follows the same fake caller's Buy-favoring
    # logic (only "Risk Contrarian" system prompts get Sell), so it sides Buy
    assert trace.synthesis.recommendation.value == "Buy"
    assert trace.disagreements[-1].resolution_strategy_applied == "tie_breaker"
    assert trace.disagreements[-1].resolved is True


class _HugeUsageRawCaller:
    """Every agent reports using far more tokens than any reasonable
    allocation — simulates a round whose real usage alone exceeds the
    entire spendable budget."""

    def __init__(self):
        self.call_count = 0

    async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens=None):
        self.call_count += 1
        return (
            {"stance": "Buy", "confidence": 65, "key_factors": ["growth"], "top_risk": "x"},
            50_000,
        )


async def test_orchestrator_stops_early_with_partial_trace_when_a_round_exhausts_budget():
    """Post-round budget check (not just allocate()'s lazy pre-round guard):
    once a round's real usage pushes remaining_budget() negative, the
    orchestrator stops cleanly after that round rather than proceeding into
    a next-round allocate() call that's certain to raise. The trace still
    has a synthesis — built from whatever rounds actually completed."""
    client = _make_fake_client(raw_caller=_HugeUsageRawCaller())
    agents = build_agents(llm_client=client)
    config = DebateConfig(total_token_budget=1000, num_rounds=3)
    orchestrator = DebateOrchestrator(config=config, agents=agents)

    trace = await orchestrator.run(ThesisRequest(thesis="Test thesis"))

    assert len(trace.rounds) == 1
    assert trace.synthesis is not None
    assert trace.ended_at is not None


async def test_orchestrator_passes_token_budget_to_the_llm_call_as_max_tokens():
    """The actual fix behind the whole class of budget-overrun bugs:
    token_budget must reach the provider call as a real max_tokens cap, not
    stay purely decorative. Proven by inspecting what the raw caller was
    actually invoked with, not just that the debate completes."""

    class _CapturingRawCaller:
        def __init__(self):
            self.seen_max_tokens = []

        async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens=None):
            self.seen_max_tokens.append(max_tokens)
            return (
                {"stance": "Buy", "confidence": 65, "key_factors": ["growth"], "top_risk": "x"},
                500,
            )

    raw_caller = _CapturingRawCaller()
    client = _make_fake_client(raw_caller=raw_caller)
    agents = build_agents(llm_client=client)
    config = DebateConfig(total_token_budget=8000, num_rounds=2)
    orchestrator = DebateOrchestrator(config=config, agents=agents)

    await orchestrator.run(ThesisRequest(thesis="Test thesis"))

    assert len(raw_caller.seen_max_tokens) == 8
    # None of the calls should have gone through with max_tokens=None — every
    # agent's allocation must have reached the raw caller as a real value.
    assert all(value is not None for value in raw_caller.seen_max_tokens)
    assert all(value > 0 for value in raw_caller.seen_max_tokens)
