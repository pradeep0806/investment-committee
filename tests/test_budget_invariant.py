"""Property-based proof of the invariants BudgetGate exists to enforce:
cumulative tokens actually spent across a debate can never exceed the
configured total_budget, and a failed call never leaks its reservation —
not "usually," not "in the example cases we thought to write," but across
randomized numbers of agents, rounds, per-call token requests, and simulated
call failures (interview follow-up: depth over breadth on the budget gate
specifically).

Hypothesis's `@given` doesn't run an async test body directly; each test
here is a plain sync function that drives the async BudgetGate calls via
asyncio.run(...) internally, once per generated example — this needs no new
pytest-asyncio/hypothesis integration dependency, just the existing asyncio
stdlib.

Design note on the raw-caller test double: it deliberately computes
tokens_used/failure as a *pure function of the max_tokens it's actually
invoked with* (via a small deterministic hash), rather than popping from a
pre-built list indexed by "the Nth call in call_plan." An earlier version
used a pre-built per-call script list and it silently desynced the moment
any call was refused by the gate before ever reaching the raw caller
(BudgetGate.call() makes no call at all on refusal) — the next *admitted*
call would then pop the *refused* call's scripted entry instead of its own,
corrupting the test's own oracle, not BudgetGate. A pure function keyed off
the actual argument the gate passes in has no such state to desync.

A second, separate discovery while building this (not the same as the
already-documented prompt-size overrun in structured_output.py): any
max_tokens BudgetGate reserves below `MIN_MAX_TOKENS` (256) is clamped
*upward* to 256 by call_structured before it ever reaches the provider —
so a call reserving as little as 1 token still tells the provider it may
generate up to 256, and real usage can land anywhere in that range,
independent of what was actually reserved. This is a distinct, wider
overrun channel than the prompt-size one, and it means "tokens_used stays
within its own reservation" is not achievable by construction for any
max_tokens < MIN_MAX_TOKENS — the codebase's own structured_output.py
guarantees exactly the opposite below that floor. The "stays within
reservation" tests below are scoped to max_tokens >= MIN_MAX_TOKENS
accordingly, and the sub-256 interaction is called out explicitly rather
than silently avoided.
"""

from __future__ import annotations

import asyncio

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import BaseModel

from committee.llm.client import LLMClient
from committee.llm.structured_output import MIN_MAX_TOKENS
from committee.orchestration.budget_gate import BudgetGate
from committee.orchestration.budget_manager import BudgetExhaustedError


class _Schema(BaseModel):
    stance: str
    confidence: int
    key_factors: list[str]
    evidence: list[str]
    top_risk: str


def _deterministic_outcome(max_tokens: int, salt: int, overrun_allowed: bool) -> tuple[bool, int]:
    """Pure function of (max_tokens, salt): returns (should_fail, tokens_used)
    with no external state, so it can never desync from which calls the gate
    actually admits — every admitted call independently recomputes its own
    outcome from its own max_tokens, nothing is consumed/popped."""
    bucket = (max_tokens * 2654435761 + salt) % 5
    should_fail = bucket == 0
    if should_fail:
        return True, 0
    if overrun_allowed and bucket == 1:
        # Legitimate, documented overrun: usage above what was requested.
        return False, max_tokens + (bucket + salt) % 50 + 1
    # Usage at or below what was requested.
    tokens_used = max_tokens - (bucket + salt) % max(max_tokens, 1)
    return False, max(tokens_used, 0)


class _DeterministicRawCaller:
    """Every call's outcome is derived purely from the max_tokens it's
    invoked with (see _deterministic_outcome) — no shared list to pop from,
    so a call the gate refuses before reaching this caller at all can never
    desync a later, different call's outcome."""

    def __init__(self, salt: int, overrun_allowed: bool = False):
        self._salt = salt
        self._overrun_allowed = overrun_allowed
        self.calls_made = 0

    async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens=None):
        self.calls_made += 1
        should_fail, tokens_used = _deterministic_outcome(max_tokens, self._salt, self._overrun_allowed)
        if should_fail:
            raise ConnectionError("simulated transport failure — no usage, nothing generated")
        return (
            {
                "stance": "Buy",
                "confidence": 50,
                "key_factors": ["x"],
                "evidence": ["y"],
                "top_risk": "z",
            },
            tokens_used,
        )


def _make_gate(total_budget: int, raw_caller) -> BudgetGate:
    client = LLMClient.__new__(LLMClient)
    client.provider = "fake"
    client.model = "fake-model"
    client.api_key = "fake-key"
    client.timeout_seconds = 60
    client.max_retries = 1
    client.retry_backoff_seconds = 0
    client.fallback_provider = None
    client.fallback_model = None
    client.fallback_max_retries = 0
    client._fallback_raw_caller = None
    client._raw_caller = raw_caller
    return BudgetGate(llm_client=client, total_budget=total_budget)


async def _run_call_plan(gate: BudgetGate, max_tokens_plan: list[int]) -> int:
    """Drives `gate.call()` once per entry in `max_tokens_plan`, tolerating
    both expected outcomes (BudgetExhaustedError on refusal,
    ConnectionError on a simulated transport failure), and returns the sum
    of tokens_used across every call that actually succeeded."""
    total_successfully_used = 0
    for max_tokens in max_tokens_plan:
        try:
            _result, tokens_used, _provider = await gate.call(
                system_prompt="sys",
                user_prompt="user",
                response_model=_Schema,
                max_tokens=max_tokens,
            )
            total_successfully_used += tokens_used
        except (BudgetExhaustedError, ConnectionError):
            pass
    return total_successfully_used


# For admission-control tests, which only care about the reserve/refuse
# decision — any positive max_tokens is fair game, including values below
# MIN_MAX_TOKENS.
_any_max_tokens_plan_strategy = st.lists(
    st.integers(min_value=1, max_value=2_000), min_size=1, max_size=40
)

# For tests that assert real tokens_used stays within its own reservation —
# only valid for max_tokens >= MIN_MAX_TOKENS, since call_structured clamps
# anything smaller up to that floor before ever calling the provider (see
# module docstring). Using a value below the floor here would be asserting
# something the codebase itself doesn't guarantee, not testing BudgetGate.
_bounded_max_tokens_plan_strategy = st.lists(
    st.integers(min_value=MIN_MAX_TOKENS, max_value=MIN_MAX_TOKENS + 5_000), min_size=1, max_size=40
)


@settings(max_examples=150, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    total_budget=st.integers(min_value=1, max_value=50_000),
    max_tokens_plan=_any_max_tokens_plan_strategy,
    salt=st.integers(min_value=0, max_value=10_000),
)
def test_gate_never_admits_a_call_whose_request_exceeds_remaining_at_that_moment(
    total_budget, max_tokens_plan, salt
):
    """The invariant BudgetGate actually implements (see budget_gate.py's own
    docstring): reservation happens *before* the call, based on requested
    max_tokens, and is refused outright — no network call at all — the
    moment it would exceed what's left *at that instant* (which moves as
    earlier calls succeed, fail-and-refund, or under-use-and-credit-back).
    Proven directly by checking, at the moment of every single call, that
    the admit/refuse decision was consistent with remaining just before it —
    not by summing requested amounts across the whole plan, which would be
    wrong: a call that uses less than it reserved legitimately frees that
    surplus for a later call to reuse (see the surplus-credit-back logic in
    budget_gate.py), so reservations are not cumulative-and-permanent."""
    raw_caller = _DeterministicRawCaller(salt, overrun_allowed=False)
    gate = _make_gate(total_budget, raw_caller)

    async def _run_and_check():
        for max_tokens in max_tokens_plan:
            remaining_before = gate.remaining
            try:
                await gate.call(
                    system_prompt="sys",
                    user_prompt="user",
                    response_model=_Schema,
                    max_tokens=max_tokens,
                )
                # Admitted — only correct if it didn't exceed what was
                # remaining the instant before this call reserved.
                assert max_tokens <= remaining_before
            except BudgetExhaustedError:
                # Refused — only correct once the request would have
                # exceeded remaining_before.
                assert max_tokens > remaining_before
            except ConnectionError:
                pass

    asyncio.run(_run_and_check())
    # remaining is NOT asserted >= 0 here: for any max_tokens below
    # MIN_MAX_TOKENS, call_structured itself clamps the request the
    # provider actually sees upward regardless of what this gate reserved
    # (see structured_output.py's module docstring) — so a call reserving
    # e.g. 1 token can legitimately report usage up to MIN_MAX_TOKENS, and
    # BudgetGate.call()'s _debit_overrun correctly reflects that as a
    # negative `remaining` rather than silently absorbing it (a real
    # accounting bug fixed in the same session: _remaining previously was
    # only ever decremented by the *reserved* amount, never by real usage
    # beyond it). The property this test actually proves — every admission
    # decision was correct given remaining at that moment — holds
    # regardless of what remaining looks like afterward.


@settings(max_examples=150, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    total_budget=st.integers(min_value=1, max_value=50_000),
    max_tokens_plan=_bounded_max_tokens_plan_strategy,
    salt=st.integers(min_value=0, max_value=10_000),
)
def test_cumulative_usage_never_exceeds_budget_when_every_call_stays_within_its_reservation(
    total_budget, max_tokens_plan, salt
):
    """The stronger, intuitive form of the invariant — cumulative *actual*
    tokens_used never exceeds total_budget, and BudgetGate's own bookkeeping
    (`remaining`) exactly matches total_budget minus what was really spent —
    genuinely holds whenever every individual call's usage stays within
    what it reserved (tokens_used <= max_tokens) AND max_tokens is at least
    MIN_MAX_TOKENS (below that floor, call_structured itself clamps the
    request upward — see module docstring — so "stays within its own
    reservation" isn't a guarantee the codebase makes at all down there).
    This is the common case in practice; the documented overrun exception
    is proven separately below since mixing the two would just reduce to
    that already-proven,
    already-explained edge case."""
    raw_caller = _DeterministicRawCaller(salt, overrun_allowed=False)
    gate = _make_gate(total_budget, raw_caller)

    total_successfully_used = asyncio.run(_run_call_plan(gate, max_tokens_plan))

    assert total_successfully_used <= total_budget
    assert gate.remaining >= 0
    assert gate.remaining == total_budget - total_successfully_used


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    total_budget=st.integers(min_value=1, max_value=50_000),
    max_tokens_plan=_bounded_max_tokens_plan_strategy,
    salt=st.integers(min_value=0, max_value=10_000),
)
def test_a_single_calls_overrun_is_bounded_and_never_lets_the_gate_admit_beyond_it(
    total_budget, max_tokens_plan, salt
):
    """The documented, accepted exception to the intuitive invariant above:
    a single admitted call's actual tokens_used can legitimately exceed its
    own max_tokens (the prompt-size-driven overrun described in
    structured_output.py, metered via budget_overrun_tokens_total) — found
    via Hypothesis on an earlier version of this test suite, not
    anticipated up front. Scoped to max_tokens >= MIN_MAX_TOKENS so this
    stays isolated to *that* overrun mechanism specifically, distinct from
    the separate sub-256 clamp interaction documented in the module
    docstring (mixing the two would make it unclear which mechanism any
    given failure was actually exercising). What must still hold even then:
    the *admission*
    decision for every call was correct given remaining at that moment (the
    gate had no way to know about the coming overrun in advance — it isn't
    supposed to), and remaining never becomes more negative than the worst
    single overrun actually reported, i.e. an overrun on one call can't
    cascade into unboundedly admitting further calls it shouldn't."""
    raw_caller = _DeterministicRawCaller(salt, overrun_allowed=True)
    gate = _make_gate(total_budget, raw_caller)

    async def _run_and_check():
        max_single_overrun = 0
        for max_tokens in max_tokens_plan:
            remaining_before = gate.remaining
            try:
                _result, tokens_used, _provider = await gate.call(
                    system_prompt="sys",
                    user_prompt="user",
                    response_model=_Schema,
                    max_tokens=max_tokens,
                )
                assert max_tokens <= remaining_before
                overrun = max(tokens_used - max_tokens, 0)
                max_single_overrun = max(max_single_overrun, overrun)
            except BudgetExhaustedError:
                assert max_tokens > remaining_before
            except ConnectionError:
                pass
        return max_single_overrun

    max_single_overrun = asyncio.run(_run_and_check())

    # remaining can only ever go as far negative as the single worst
    # overrun seen (each call's overrun independently drags remaining down
    # by at most its own overrun, on top of an admission that was correct
    # at the time) — not unboundedly, and not cumulatively across many
    # calls' overruns compounding into something unrelated to any one
    # call's actual reported usage.
    assert gate.remaining >= -max_single_overrun


@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    total_budget=st.integers(min_value=1, max_value=10_000),
    num_agents=st.integers(min_value=1, max_value=8),
    num_rounds=st.integers(min_value=1, max_value=4),
    salt=st.integers(min_value=0, max_value=10_000),
)
def test_invariant_holds_across_randomized_agent_and_round_counts(
    total_budget, num_agents, num_rounds, salt
):
    """Same admission-control invariant as
    test_gate_never_admits_a_call_whose_request_exceeds_remaining_at_that_moment,
    generated from a more debate-shaped angle (explicit agents x rounds
    grid, per the task's own framing) rather than an arbitrary flat list —
    each simulated agent-round requests an even baseline allocation,
    mirroring BudgetManager.allocate's real per-round strategy.

    Deliberately does *not* assert total_successfully_used <= total_budget
    here: with many agents/rounds splitting a modest total_budget, the
    per-agent baseline routinely falls below MIN_MAX_TOKENS (256) — a
    realistic, common case for a real debate, not a contrived one — and the
    sub-256 clamp interaction documented in the module docstring means real
    usage can then legitimately exceed the naive baseline-times-call-count
    sum. That's a property of call_structured's own floor, not something
    this test should paper over by only using large budgets. What this test
    proves instead is the same thing the dedicated admission-control test
    proves: every individual call's admit/refuse decision was consistent
    with what was actually remaining at that moment."""
    baseline = max(total_budget // (num_agents * num_rounds), 1)
    max_tokens_plan = [baseline] * (num_agents * num_rounds)

    raw_caller = _DeterministicRawCaller(salt, overrun_allowed=False)
    gate = _make_gate(total_budget, raw_caller)

    async def _run_and_check():
        for max_tokens in max_tokens_plan:
            remaining_before = gate.remaining
            try:
                await gate.call(
                    system_prompt="sys",
                    user_prompt="user",
                    response_model=_Schema,
                    max_tokens=max_tokens,
                )
                assert max_tokens <= remaining_before
            except BudgetExhaustedError:
                assert max_tokens > remaining_before
            except ConnectionError:
                pass

    asyncio.run(_run_and_check())


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    total_budget=st.integers(min_value=1, max_value=20_000),
    failing_max_tokens=st.lists(st.integers(min_value=1, max_value=5_000), min_size=1, max_size=15),
)
def test_every_failed_call_fully_refunds_its_reservation(total_budget, failing_max_tokens):
    """Isolates the refund-on-failure guarantee explicitly (Core C's
    "including cases with simulated call failures"): a sequence of calls
    that *every one* fails before reporting any usage must leave
    BudgetGate.remaining exactly where it started — a reservation that's
    released must be released in full, not partially, regardless of how
    many failures happen in a row or what each one's max_tokens request
    was."""

    class _AlwaysFailingRawCaller:
        async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens=None):
            raise ConnectionError("simulated transport failure — no usage, nothing generated")

    gate = _make_gate(total_budget, _AlwaysFailingRawCaller())
    starting_remaining = gate.remaining

    async def _run_all_calls():
        for max_tokens in failing_max_tokens:
            try:
                await gate.call(
                    system_prompt="sys",
                    user_prompt="user",
                    response_model=_Schema,
                    max_tokens=max_tokens,
                )
                raise AssertionError("this call was scripted to fail, but it succeeded")
            except ConnectionError:
                pass
            except BudgetExhaustedError:
                # Only possible if max_tokens itself exceeded the (never-
                # spent, since every prior call also failed and refunded)
                # starting budget — still consistent with "nothing was ever
                # actually spent."
                pass

    asyncio.run(_run_all_calls())

    assert gate.remaining == starting_remaining
