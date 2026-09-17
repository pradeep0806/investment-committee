"""BudgetGate: the single structural chokepoint between any caller and the
underlying LLMClient.

The original bug (CLAUDE.md-documented, since fixed) was that token budget was
*computed* but never actually enforced at the point the LLM was called —
`max_tokens` was accepted as a parameter and simply never reached the API.
That's fixed today (see llm/structured_output.py), but the fix was
procedural: any code path holding an `LLMClient` reference can still call
`.call(...)` with `max_tokens=None` or with a number that overspends, and
nothing stops it. There is no structural reason a future call site couldn't
reintroduce the same class of bug.

BudgetGate closes that off entirely. It is constructed as the *sole* holder
of the LLMClient reference; every agent (and the tie-breaker) receives a gate,
never a raw client, so `.call()` is simply not reachable except through here.
`call()`:
  1. requires an explicit `max_tokens` (no bypassing enforcement via None),
  2. atomically reserves that many tokens from the remaining budget *before*
     issuing the call — reservation happens or the call doesn't,
  3. refuses (raises BudgetExhaustedError, no network call made) if the
     request would exceed what's left.

Reservation is deliberately based on the *requested* max_tokens, not actual
usage — actual usage is only known after the call returns, by which point a
refusal would be too late to mean anything. If the call uses less than
reserved, the surplus is credited back so a conservative reservation doesn't
permanently waste headroom; it never re-credits more than was reserved, so
the reservation is still the hard ceiling at call time.

Two-tier enforcement: an in-process asyncio.Lock always guards the local
`_remaining` cache (correct and sufficient for a single debate inside a
single process). When a `budget_store` + `run_id` are also given, every
reservation/release is additionally routed through BudgetStore's atomic
Mongo `$inc` (storage/budget_store.py) — the database becomes the actual
source of truth, so two debates (or two processes, e.g. after a crash and
resume) touching the same run_id's budget can never both succeed past what's
really left. Without a budget_store (matching the rest of this codebase's
"Mongo is best-effort, the debate must still run" stance), the gate still
fully enforces budget — just scoped to this one process.
"""

from __future__ import annotations

import asyncio
from typing import Protocol, TypeVar

from pydantic import BaseModel

from committee.llm.client import LLMClient
from committee.llm.structured_output import LLMValidationError
from committee.orchestration.budget_manager import BudgetExhaustedError

ModelT = TypeVar("ModelT", bound=BaseModel)


class _AtomicBudgetStore(Protocol):
    """The subset of storage/budget_store.py's BudgetStore that BudgetGate
    actually needs — a Protocol (like TraceStore, ConflictResolutionStrategy
    elsewhere in this codebase) so this module doesn't need a hard import of
    the Mongo-specific implementation just to type-check against it."""

    async def initialize(self, run_id: str, total_budget: int) -> None: ...

    async def reserve(self, run_id: str, amount: int) -> bool: ...

    async def release(self, run_id: str, amount: int) -> None: ...

    async def debit_overrun(self, run_id: str, amount: int) -> None: ...


class BudgetGate:
    def __init__(
        self,
        llm_client: LLMClient,
        total_budget: int,
        budget_store: _AtomicBudgetStore | None = None,
        run_id: str | None = None,
    ):
        self._llm_client = llm_client
        self._remaining = total_budget
        self._total_budget = total_budget
        # Every reservation + refund is one atomic step from the caller's
        # perspective; asyncio.Lock is enough for same-process concurrency —
        # budget_store (below) extends the same atomicity guarantee across
        # processes/debates via Mongo.
        self._lock = asyncio.Lock()
        self._budget_store = budget_store
        self._run_id = run_id

    def bind_run(self, run_id: str, budget_store: _AtomicBudgetStore | None) -> None:
        """Called once by the orchestrator as soon as a debate's run_id is
        known (a fresh uuid4, or the resumed run_id) — BudgetGate is
        constructed by orchestrator_factory.py before any run_id exists, so
        the atomic-store binding necessarily happens in two steps: the store
        connection at construction time, the run_id once a debate actually
        starts. A no-op if budget_store is None (matching the rest of this
        codebase's Mongo-is-optional stance)."""
        self._run_id = run_id
        self._budget_store = budget_store

    @property
    def remaining(self) -> int:
        return self._remaining

    @property
    def model(self) -> str:
        """Read-only passthrough for observability (e.g. MLflow run params)
        — metadata, not a call path, so exposing it doesn't reopen the
        bypass this gate exists to close."""
        return self._llm_client.model

    async def _initialize_store(self) -> None:
        if self._budget_store is not None and self._run_id is not None:
            await self._budget_store.initialize(self._run_id, self._total_budget)

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ModelT],
        max_tokens: int,
        max_retries: int | None = None,
    ) -> tuple[ModelT, int, str]:
        """The only path to LLMClient.call(). `max_tokens` is required (not
        optional) — a caller cannot opt out of enforcement by omitting it.
        Reserves `max_tokens` before the call; raises BudgetExhaustedError
        and makes no call at all if that would overspend the remaining
        budget. Reconciles the reservation against actual tokens_used once
        the call returns (or releases it fully if the call raised).

        Returns (result, tokens_used, provider_used) — provider_used is
        whichever provider (primary or fallback) actually served the call,
        passed through unchanged from LLMClient.call(). This gate's
        reservation/deduction happens exactly once per call regardless of
        which provider ends up serving it: LLMClient's own retry-then-
        fallback logic (client.py) runs entirely *inside* the single
        `self._llm_client.call(...)` below, so a fallback never gets its own
        separate reservation — there is no path through this gate that
        skips or double-charges budget enforcement based on provider
        choice. Token accounting is a documented approximation across
        providers (see client.py/README): tokens from a fallback provider
        are deducted 1:1 against the same budget as the primary's, even
        though different providers/models don't cost the same per token —
        provider_used is exactly what lets a reviewer see when that
        approximation was in effect for a given call."""
        await self._reserve(max_tokens)

        try:
            result, tokens_used, provider_used = await self._llm_client.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=response_model,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )
        except LLMValidationError as exc:
            # Real bug found via a live debate against a weak local model
            # (qwen3.5:9b via Ollama): every attempt made before giving up
            # is a real call against the real provider, and
            # structured_output.py's retry loop already tracks that spend
            # (exc.total_tokens_used) — releasing the *full* max_tokens
            # reservation here, as the blanket `except Exception` below
            # used to do, silently credited back budget for tokens that
            # were genuinely spent. Release only the unused remainder.
            unused = max(max_tokens - exc.total_tokens_used, 0)
            if unused:
                await self._release(unused)
            overrun = max(exc.total_tokens_used - max_tokens, 0)
            if overrun:
                await self._debit_overrun(overrun)
            raise
        except Exception:
            # No structured_output.py accounting to reconcile against here
            # (a transport-level failure this module itself doesn't wrap,
            # or a bug) — assume zero spend, matching this gate's
            # historical behavior for every exception type before
            # LLMValidationError's total_tokens_used existed.
            await self._release(max_tokens)
            raise

        # Credit back any surplus between what was reserved and what was
        # actually used...
        surplus = max(max_tokens - tokens_used, 0)
        if surplus:
            await self._release(surplus)
        # ...or, if actual usage exceeded the reservation (the documented
        # prompt-size asymmetry, or structured_output.py's own
        # RETRY_BUDGET_MULTIPLIER-bounded retry overrun), debit the
        # difference from `remaining` instead of silently absorbing it.
        # Real bug found alongside the one above: `_remaining` was
        # previously only ever decremented by the *reserved* max_tokens,
        # never by real usage beyond that — so `remaining` overstated what
        # was actually left the moment any call overran its own
        # reservation, which structured_output.py's own documented
        # prompt-size asymmetry already made possible even before this
        # session's retry-budget change. This can drive `remaining`
        # negative — deliberately: the reservation-before-call check is
        # what prevents a *new* call from being admitted past the budget
        # (see _reserve), not a promise that `remaining` never dips below
        # zero after the fact once a single call's real cost exceeds what
        # it reserved.
        overrun = max(tokens_used - max_tokens, 0)
        if overrun:
            await self._debit_overrun(overrun)

        return result, tokens_used, provider_used

    async def _reserve(self, amount: int) -> None:
        async with self._lock:
            if amount > self._remaining:
                raise BudgetExhaustedError(
                    f"Requested max_tokens={amount} exceeds remaining gate budget "
                    f"{self._remaining}; refusing to call the LLM."
                )
            if self._budget_store is not None and self._run_id is not None:
                await self._initialize_store()
                ok = await self._budget_store.reserve(self._run_id, amount)
                if not ok:
                    # The DB-level guard is the real source of truth once a
                    # store is wired in — a local cache that thinks there's
                    # room but the atomic store disagrees (e.g. another
                    # process/debate reserved first) must defer to the store.
                    raise BudgetExhaustedError(
                        f"Requested max_tokens={amount} exceeds remaining budget for "
                        f"run_id={self._run_id!r} per the atomic budget store; refusing "
                        "to call the LLM."
                    )
            self._remaining -= amount

    async def _release(self, amount: int) -> None:
        async with self._lock:
            self._remaining += amount
            if self._budget_store is not None and self._run_id is not None:
                await self._budget_store.release(self._run_id, amount)

    async def _debit_overrun(self, amount: int) -> None:
        """Deducts real usage beyond what a call reserved — the inverse of
        `_release`'s surplus credit-back. Can drive `_remaining` (and the
        atomic store's counter) negative; that's the correct reflection of
        "more was actually spent than the budget had left," not a bug to
        clamp away. `_reserve`'s own admission check is what prevents a
        *future* call from starting once the budget is genuinely gone —
        this method only ever runs after a call has already happened, so
        there is nothing left to refuse at this point."""
        async with self._lock:
            self._remaining -= amount
            if self._budget_store is not None and self._run_id is not None:
                await self._budget_store.debit_overrun(self._run_id, amount)
