import pytest

from committee.observability.metrics import budget_overrun_tokens_total
from committee.orchestration.budget_manager import BudgetExhaustedError, BudgetManager


def test_reserve_pool_is_carved_out_up_front():
    manager = BudgetManager(total_token_budget=10_000, num_rounds=2, num_agents=4)
    assert manager.reserve_pool == 1000
    assert manager.spendable_budget == 9000


def test_explore_mode_allocates_even_split():
    manager = BudgetManager(total_token_budget=8000, num_rounds=2, num_agents=4)
    allocation = manager.allocate(round=1, agent_ids=["a", "b", "c", "d"], mode="explore")
    assert len(set(allocation.values())) == 1
    assert sum(allocation.values()) <= manager.remaining_budget() + sum(allocation.values())


def test_exploit_mode_never_starves_non_contested_agents_below_viable_floor():
    """Real bug found via a live debate against qwen3.5:9b on Ollama: by a
    later round, exploit mode's 2.5x weighting toward contested agents drove
    non-contested agents down to 723 tokens each — above the historical
    '1 token' floor, so allocate() never raised, but still not enough for
    this model to produce a valid tool call within RETRY_BUDGET_MULTIPLIER's
    retry ceiling, so those agents were excluded every round. The round's
    *total* budget was enough for an even split above the floor; the 2.5x
    reallocation itself was what starved them. Every agent — contested or
    not — must get at least min_viable_allocation. Uses an explicit,
    generously large min_viable_allocation here (2048, matching what the
    live debate that exposed this bug actually needed for qwen3.5:9b) since
    the class default (256) is deliberately modest/provider-agnostic and
    wouldn't have reproduced the original failure at these budget/round
    numbers — this test is about the reallocation math, not the default."""
    manager = BudgetManager(
        total_token_budget=40_000, num_rounds=3, num_agents=4, min_viable_allocation=2048
    )
    agent_ids = ["fundamentals", "market_sentiment", "risk_contrarian", "macro_context"]

    # Simulate two prior rounds of heavy spend (mirrors the live trace: by
    # round 3, remaining budget was thin enough that an even split would
    # have landed each agent well under min_viable_allocation before any
    # exploit-mode skew was even applied).
    for round_num in (1, 2):
        allocation = manager.allocate(
            round=round_num, agent_ids=agent_ids, mode="exploit", contested_agents=["market_sentiment"]
        )
        for agent_id, tokens_allocated in allocation.items():
            manager.record_actual_usage(
                round=round_num,
                agent_id=agent_id,
                tokens_allocated=tokens_allocated,
                tokens_used=tokens_allocated,
                mode="exploit",
            )

    round3_allocation = manager.allocate(
        round=3, agent_ids=agent_ids, mode="exploit", contested_agents=["market_sentiment"]
    )

    for agent_id, tokens_allocated in round3_allocation.items():
        assert tokens_allocated >= manager.min_viable_allocation, (
            f"{agent_id} allocated {tokens_allocated}, below the viable floor "
            f"of {manager.min_viable_allocation} — would be set up to fail, not "
            "just given a smaller-but-workable budget"
        )
    # Contested agent still gets preferential treatment over non-contested
    # ones, even once the floor guarantee is in play.
    assert round3_allocation["market_sentiment"] > round3_allocation["fundamentals"]


def test_exploit_mode_reallocates_more_to_contested_agents():
    manager = BudgetManager(total_token_budget=8000, num_rounds=2, num_agents=4)
    allocation = manager.allocate(
        round=1,
        agent_ids=["a", "b", "c", "d"],
        mode="exploit",
        contested_agents=["a"],
    )
    assert allocation["a"] > allocation["b"]
    assert allocation["a"] > allocation["c"]
    assert allocation["a"] > allocation["d"]


def test_worst_case_allocation_never_exceeds_total_budget():
    """Hard invariant (CLAUDE.md §9): simulate the worst case — every round in
    exploit mode with maximal reallocation to a single contested agent — and
    confirm cumulative actual usage never exceeds total_token_budget, even
    when every agent spends its full allocation."""
    total_budget = 20_000
    num_rounds = 3
    num_agents = 4
    manager = BudgetManager(total_token_budget=total_budget, num_rounds=num_rounds, num_agents=num_agents)

    agent_ids = ["fundamentals", "market_sentiment", "risk_contrarian", "macro_context"]
    cumulative_used = 0
    for round_num in range(1, num_rounds + 1):
        allocation = manager.allocate(
            round=round_num,
            agent_ids=agent_ids,
            mode="exploit",
            contested_agents=["risk_contrarian"],
        )
        for agent_id, tokens_allocated in allocation.items():
            manager.record_actual_usage(
                round=round_num,
                agent_id=agent_id,
                tokens_allocated=tokens_allocated,
                tokens_used=tokens_allocated,  # worst case: full spend of allocation
                mode="exploit",
            )
            cumulative_used += tokens_allocated

    assert cumulative_used <= manager.spendable_budget
    assert cumulative_used <= total_budget


def test_reserve_draw_within_reserve_succeeds():
    manager = BudgetManager(total_token_budget=10_000, num_rounds=2, num_agents=4)
    manager.draw_from_reserve(500)
    assert manager.remaining_reserve() == 500


def test_reserve_draw_exceeding_reserve_raises():
    manager = BudgetManager(total_token_budget=10_000, num_rounds=2, num_agents=4)
    with pytest.raises(BudgetExhaustedError):
        manager.draw_from_reserve(2000)


def test_allocate_raises_when_budget_cannot_cover_floor():
    manager = BudgetManager(total_token_budget=10, num_rounds=2, num_agents=4)
    # Drain the spendable budget entirely.
    manager.record_actual_usage(
        round=1, agent_id="a", tokens_allocated=9, tokens_used=9, mode="explore"
    )
    with pytest.raises(BudgetExhaustedError):
        manager.allocate(round=2, agent_ids=["a", "b", "c", "d"], mode="explore")


def test_ledger_records_allocated_and_actual_usage_separately():
    manager = BudgetManager(total_token_budget=8000, num_rounds=2, num_agents=4)
    manager.record_actual_usage(
        round=1, agent_id="a", tokens_allocated=1000, tokens_used=850, mode="explore"
    )
    entry = manager.ledger[0]
    assert entry.tokens_allocated == 1000
    assert entry.tokens_used == 850
    assert manager.remaining_budget() == manager.spendable_budget - 850


def test_allocation_shrinks_for_later_rounds_after_an_earlier_round_overspends():
    """Graceful degradation: real LLM calls routinely use more tokens than
    their allocation (token_budget is advisory to the model, not a hard cap
    on its response). If round 1 overspends its baseline, round 2's baseline
    must shrink to compensate rather than the debate crashing outright — the
    real bug this test guards against: a live Gemini call using ~1243 tokens
    against a ~675-token baseline caused BudgetExhaustedError on round 2
    before this dynamic recomputation was added. Uses a large enough total
    budget that round 2 still has genuinely enough left for a viable
    (>= min_viable_allocation/agent) allocation after the overspend — the
    "not enough left at all" case is covered separately below."""
    manager = BudgetManager(total_token_budget=200_000, num_rounds=2, num_agents=4)

    round1_allocation = manager.allocate(round=1, agent_ids=["a", "b", "c", "d"], mode="explore")
    round1_baseline = next(iter(round1_allocation.values()))

    # Every agent overspends its round-1 allocation (mirrors the real ~1.8x
    # overrun seen from a live Gemini call), but not so much that round 1
    # alone exhausts the whole spendable budget.
    overspend_per_agent = int(round1_baseline * 1.8)
    for agent_id in ["a", "b", "c", "d"]:
        manager.record_actual_usage(
            round=1,
            agent_id=agent_id,
            tokens_allocated=round1_allocation[agent_id],
            tokens_used=overspend_per_agent,
            mode="explore",
        )

    # Round 2 must not raise, and its baseline should be smaller than round 1's
    # (there's genuinely less budget left than an even split would assume).
    round2_allocation = manager.allocate(round=2, agent_ids=["a", "b", "c", "d"], mode="explore")
    round2_baseline = next(iter(round2_allocation.values()))
    assert round2_baseline < round1_baseline
    assert round2_baseline >= manager.min_viable_allocation


def test_allocate_raises_rather_than_give_an_agent_an_unviable_allocation():
    """Real bug found via a live debate against qwen3.5:9b on Ollama: with
    the old '1 token/agent' floor, a round with genuinely little budget left
    would still 'succeed' at allocate() time, only to have every agent
    excluded later because the allocation was structurally unable to
    produce a valid tool call. Once remaining budget can no longer afford
    min_viable_allocation for every agent, allocate() must raise
    immediately (a debate stopping cleanly with fewer, valid rounds) rather
    than silently hand out an allocation that sets every agent up to fail."""
    manager = BudgetManager(total_token_budget=4_000, num_rounds=2, num_agents=4)

    round1_allocation = manager.allocate(round=1, agent_ids=["a", "b", "c", "d"], mode="explore")
    round1_baseline = next(iter(round1_allocation.values()))

    # A much heavier overspend than the "shrinks but survives" test above —
    # this time round 2 genuinely doesn't have enough left for a viable
    # allocation (below the default min_viable_allocation, 256/agent), and
    # must say so rather than proceed anyway.
    overspend_per_agent = int(round1_baseline * 3.5)
    for agent_id in ["a", "b", "c", "d"]:
        manager.record_actual_usage(
            round=1,
            agent_id=agent_id,
            tokens_allocated=round1_allocation[agent_id],
            tokens_used=overspend_per_agent,
            mode="explore",
        )

    with pytest.raises(BudgetExhaustedError):
        manager.allocate(round=2, agent_ids=["a", "b", "c", "d"], mode="explore")


def test_allocation_grows_for_later_rounds_after_an_earlier_round_underspends():
    """The inverse case: if round 1 underspends its baseline, round 2 should
    get a larger allocation than a fixed even-split would give it, since more
    budget is genuinely available."""
    manager = BudgetManager(total_token_budget=8000, num_rounds=2, num_agents=4)

    round1_allocation = manager.allocate(round=1, agent_ids=["a", "b", "c", "d"], mode="explore")
    round1_baseline = next(iter(round1_allocation.values()))

    underspend_per_agent = round1_baseline // 4
    for agent_id in ["a", "b", "c", "d"]:
        manager.record_actual_usage(
            round=1,
            agent_id=agent_id,
            tokens_allocated=round1_allocation[agent_id],
            tokens_used=underspend_per_agent,
            mode="explore",
        )

    round2_allocation = manager.allocate(round=2, agent_ids=["a", "b", "c", "d"], mode="explore")
    round2_baseline = next(iter(round2_allocation.values()))
    assert round2_baseline > round1_baseline


def test_record_actual_usage_increments_overrun_metric_when_usage_exceeds_allocation():
    """The single most diagnostic signal for what's actually going on when an
    agent silently uses far more than it was allocated — without this, an
    agent using 10x its allocation would be invisible except by diffing
    tokens_allocated vs tokens_used in the ledger by hand."""
    manager = BudgetManager(total_token_budget=8000, num_rounds=2, num_agents=4)
    before = budget_overrun_tokens_total.labels(agent="test-overrun-agent")._value.get()

    manager.record_actual_usage(
        round=1, agent_id="test-overrun-agent", tokens_allocated=100, tokens_used=1000, mode="explore"
    )

    after = budget_overrun_tokens_total.labels(agent="test-overrun-agent")._value.get()
    assert after - before == 900


def test_record_actual_usage_does_not_increment_overrun_metric_when_within_allocation():
    manager = BudgetManager(total_token_budget=8000, num_rounds=2, num_agents=4)
    before = budget_overrun_tokens_total.labels(agent="test-no-overrun-agent")._value.get()

    manager.record_actual_usage(
        round=1, agent_id="test-no-overrun-agent", tokens_allocated=1000, tokens_used=850, mode="explore"
    )

    after = budget_overrun_tokens_total.labels(agent="test-no-overrun-agent")._value.get()
    assert after == before


@pytest.mark.parametrize("num_agents", [4, 5, 6, 8])
def test_allocate_divides_evenly_across_whatever_agent_count_is_given(num_agents):
    """The allocator must work for however many agents a debate actually has
    (core 4 + N custom personas), not assume a fixed count — this is the
    property persona-as-data's dynamic budget allocation depends on."""
    agent_ids = [f"agent-{i}" for i in range(num_agents)]
    manager = BudgetManager(total_token_budget=20_000, num_rounds=2, num_agents=num_agents)

    allocation = manager.allocate(round=1, agent_ids=agent_ids, mode="explore")

    assert len(allocation) == num_agents
    assert len(set(allocation.values())) == 1
    assert sum(allocation.values()) <= manager.spendable_budget


@pytest.mark.parametrize("num_agents", [5, 6])
def test_exploit_mode_reallocation_scales_with_agent_count(num_agents):
    agent_ids = [f"agent-{i}" for i in range(num_agents)]
    manager = BudgetManager(total_token_budget=30_000, num_rounds=2, num_agents=num_agents)

    allocation = manager.allocate(
        round=1, agent_ids=agent_ids, mode="exploit", contested_agents=["agent-0"]
    )

    assert len(allocation) == num_agents
    for other_id in agent_ids[1:]:
        assert allocation["agent-0"] > allocation[other_id]
    assert sum(allocation.values()) <= manager.spendable_budget


def test_worst_case_allocation_never_exceeds_budget_with_six_agents():
    total_budget = 30_000
    num_rounds = 3
    num_agents = 6
    manager = BudgetManager(total_token_budget=total_budget, num_rounds=num_rounds, num_agents=num_agents)
    agent_ids = [f"agent-{i}" for i in range(num_agents)]

    cumulative_used = 0
    for round_num in range(1, num_rounds + 1):
        allocation = manager.allocate(
            round=round_num, agent_ids=agent_ids, mode="exploit", contested_agents=["agent-0"]
        )
        for agent_id, tokens_allocated in allocation.items():
            manager.record_actual_usage(
                round=round_num,
                agent_id=agent_id,
                tokens_allocated=tokens_allocated,
                tokens_used=tokens_allocated,
                mode="exploit",
            )
            cumulative_used += tokens_allocated

    assert cumulative_used <= manager.spendable_budget
    assert cumulative_used <= total_budget


def test_cumulative_usage_still_never_exceeds_budget_even_with_dynamic_rebaselining():
    """The hard invariant survives graceful degradation: no matter how
    allocations shrink or grow round to round, actual cumulative usage never
    exceeds total_token_budget."""
    manager = BudgetManager(total_token_budget=10_000, num_rounds=3, num_agents=4)
    agent_ids = ["a", "b", "c", "d"]
    cumulative_used = 0

    for round_num in range(1, 4):
        allocation = manager.allocate(round=round_num, agent_ids=agent_ids, mode="explore")
        for agent_id, tokens_allocated in allocation.items():
            # Simulate unpredictable real usage: sometimes over, sometimes under.
            actual_used = tokens_allocated + (50 if round_num == 1 else -20)
            actual_used = max(actual_used, 0)
            manager.record_actual_usage(
                round=round_num,
                agent_id=agent_id,
                tokens_allocated=tokens_allocated,
                tokens_used=actual_used,
                mode="explore",
            )
            cumulative_used += actual_used

    assert cumulative_used <= manager.spendable_budget
