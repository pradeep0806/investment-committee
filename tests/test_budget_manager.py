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
    before this dynamic recomputation was added."""
    manager = BudgetManager(total_token_budget=20_000, num_rounds=2, num_agents=4)

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
    assert round2_baseline >= 1


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
