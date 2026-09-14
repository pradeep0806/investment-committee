"""Tests BudgetStore's atomic-reservation contract against a fake Mongo
collection that faithfully implements find_one_and_update's filter+update
semantics (no real Mongo in this test environment — matches the existing
pattern of fake/double-based storage tests, see test_storage.py). The point
under test is the atomicity contract itself: a filtered find_one_and_update
is a single indivisible step, so two coroutines racing to reserve from the
same document cannot both observe "enough remaining" and overspend.
"""

import asyncio

from committee.storage.budget_store import BudgetStore


class _FakeCollection:
    """Single-document-per-run_id fake with just enough Mongo semantics to
    prove the atomicity contract: find_one_and_update evaluates its filter
    and applies its update as one step, with no await point in between where
    another coroutine could interleave."""

    def __init__(self):
        self._docs: dict[str, dict] = {}

    async def update_one(self, filter_, update, upsert=False):
        run_id = filter_["run_id"]
        if run_id not in self._docs:
            if upsert and "$setOnInsert" in update:
                self._docs[run_id] = dict(update["$setOnInsert"])
            return
        if "$inc" in update:
            for field, delta in update["$inc"].items():
                self._docs[run_id][field] = self._docs[run_id].get(field, 0) + delta

    async def find_one_and_update(self, filter_, update):
        # No `await` between the filter check and the mutation — this is
        # the property that makes it atomic under cooperative scheduling,
        # exactly mirroring what Mongo guarantees server-side for a single
        # document.
        run_id = filter_["run_id"]
        doc = self._docs.get(run_id)
        if doc is None:
            return None
        gte = filter_.get("remaining", {}).get("$gte")
        if gte is not None and doc["remaining"] < gte:
            return None
        for field, delta in update.get("$inc", {}).items():
            doc[field] = doc.get(field, 0) + delta
        return doc

    async def find_one(self, filter_):
        return self._docs.get(filter_["run_id"])


def _make_store() -> BudgetStore:
    store = BudgetStore.__new__(BudgetStore)
    store._collection = _FakeCollection()
    return store


class TestBudgetStoreAtomicity:
    async def test_initialize_creates_ledger_with_full_remaining(self):
        store = _make_store()
        await store.initialize("run-1", total_budget=1000)
        assert await store.remaining("run-1") == 1000

    async def test_initialize_does_not_reset_an_already_partially_spent_ledger(self):
        """Resume path: initialize() must be idempotent-on-existing, not a
        reset — otherwise resuming a debate would refund all previously
        spent budget."""
        store = _make_store()
        await store.initialize("run-1", total_budget=1000)
        await store.reserve("run-1", 400)
        assert await store.remaining("run-1") == 600

        await store.initialize("run-1", total_budget=1000)
        assert await store.remaining("run-1") == 600

    async def test_reserve_within_budget_succeeds_and_deducts(self):
        store = _make_store()
        await store.initialize("run-1", total_budget=1000)
        ok = await store.reserve("run-1", 300)
        assert ok is True
        assert await store.remaining("run-1") == 700

    async def test_reserve_exceeding_remaining_fails_and_deducts_nothing(self):
        store = _make_store()
        await store.initialize("run-1", total_budget=100)
        ok = await store.reserve("run-1", 500)
        assert ok is False
        assert await store.remaining("run-1") == 100

    async def test_release_credits_back(self):
        store = _make_store()
        await store.initialize("run-1", total_budget=1000)
        await store.reserve("run-1", 400)
        await store.release("run-1", 150)
        assert await store.remaining("run-1") == 750

    async def test_two_debates_deducting_concurrently_do_not_corrupt_each_others_budget(self):
        """Different run_ids are fully independent ledgers — a heavy
        concurrent reserve loop on one must never affect the other's
        remaining count."""
        store = _make_store()
        await store.initialize("debate-a", total_budget=5000)
        await store.initialize("debate-b", total_budget=5000)

        async def spend(run_id: str, amount: int, times: int):
            for _ in range(times):
                await store.reserve(run_id, amount)

        await asyncio.gather(
            spend("debate-a", 100, 30),  # 3000 total, within 5000
            spend("debate-b", 100, 30),
        )

        assert await store.remaining("debate-a") == 2000
        assert await store.remaining("debate-b") == 2000

    async def test_concurrent_reservations_within_one_debate_never_overspend(self):
        """The core concurrency guarantee: many coroutines racing to reserve
        against the SAME run_id, where the sum of all requested amounts
        exceeds the total budget, must never let cumulative deductions go
        negative or let more succeed than the budget actually allows."""
        store = _make_store()
        total_budget = 1000
        await store.initialize("run-1", total_budget=total_budget)

        per_call = 137
        num_calls = 20  # 20 * 137 = 2740, well over the 1000 budget

        async def try_reserve():
            return await store.reserve("run-1", per_call)

        results = await asyncio.gather(*(try_reserve() for _ in range(num_calls)))

        succeeded = sum(1 for r in results if r)
        remaining = await store.remaining("run-1")

        assert remaining >= 0
        assert remaining == total_budget - succeeded * per_call
        # At most floor(1000/137) = 7 reservations could ever succeed.
        assert succeeded <= total_budget // per_call

    async def test_remaining_never_goes_negative_under_heavy_contention(self):
        store = _make_store()
        await store.initialize("run-1", total_budget=50)

        async def try_reserve():
            return await store.reserve("run-1", 10)

        await asyncio.gather(*(try_reserve() for _ in range(50)))
        remaining = await store.remaining("run-1")
        assert remaining >= 0
        assert remaining % 10 == 0
