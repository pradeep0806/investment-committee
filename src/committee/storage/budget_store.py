"""BudgetStore: atomic, database-level token budget ledger, scoped by
debate (run_id).

BudgetGate's in-process reservation (asyncio.Lock) is correct for a single
debate running inside a single process, but does nothing for two debates —
or, after a crash/restart, two processes — touching the same run_id's
budget concurrently. This is the piece that makes deduction atomic at the
database level, per the task's explicit ask: a `findOneAndUpdate` with
`$inc`, scoped by debate ID, so two near-simultaneous reservations against
the same run_id can never both succeed past what's actually left.

The guard is expressed directly in the query filter (`remaining >= amount`),
not read-then-write — Mongo evaluates the filter and applies the update
atomically for a single document, so there is no window between checking
"is there enough left" and deducting it where a second writer could sneak
in. If the filter doesn't match (not enough remaining), `find_one_and_update`
returns None and no write happens at all.
"""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient


class BudgetStore:
    def __init__(self, mongo_uri: str, mongo_db: str):
        self._client: AsyncIOMotorClient = AsyncIOMotorClient(mongo_uri)
        self._collection = self._client[mongo_db]["debate_budgets"]

    async def initialize(self, run_id: str, total_budget: int) -> None:
        """Creates the ledger document for a debate if it doesn't already
        exist (upsert, not overwrite) — safe to call again on resume without
        resetting a budget that's already been partially spent."""
        await self._collection.update_one(
            {"run_id": run_id},
            {
                "$setOnInsert": {
                    "run_id": run_id,
                    "total_budget": total_budget,
                    "remaining": total_budget,
                }
            },
            upsert=True,
        )

    async def reserve(self, run_id: str, amount: int) -> bool:
        """Atomically deducts `amount` from run_id's remaining budget.
        Returns True if the reservation succeeded, False if remaining
        budget was insufficient (no write happened in that case). The
        filter's `remaining >= amount` clause and the `$inc` update are
        evaluated together by Mongo as a single atomic operation — two
        concurrent reserve() calls against the same run_id (from two
        debates, or two processes racing after a resume) cannot both
        observe "enough remaining" and overspend past zero."""
        result = await self._collection.find_one_and_update(
            {"run_id": run_id, "remaining": {"$gte": amount}},
            {"$inc": {"remaining": -amount}},
        )
        return result is not None

    async def release(self, run_id: str, amount: int) -> None:
        """Credits `amount` back — used when a reservation's call failed
        outright (nothing was spent) or used less than reserved."""
        await self._collection.update_one({"run_id": run_id}, {"$inc": {"remaining": amount}})

    async def remaining(self, run_id: str) -> int | None:
        document = await self._collection.find_one({"run_id": run_id})
        return document["remaining"] if document else None

    async def close(self) -> None:
        self._client.close()
