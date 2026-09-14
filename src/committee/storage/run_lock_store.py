"""RunLockStore: atomic, database-level cross-pod mutual exclusion for a
debate run_id.

The orchestrator's in-process `asyncio.Lock` (see orchestration/orchestrator.py's
`_active_run_locks`) is correct for a single process — it was built after a
real bug where two overlapping resume requests for the same run_id raced each
other's checkpoint/trace writes, leaving JSON and Mongo with different round
counts for the same debate. But it does nothing for two separate pods: each
runs its own Python process with its own empty lock dict, so a k8s deployment
with multiple `committee-api` replicas (see k8s/api.yaml's `replicas: 2` /
HPA) can reproduce the exact same corruption across pods instead of within
one.

This closes that gap the same way BudgetStore closes the analogous
budget-double-spend gap: a `find_one_and_update` whose filter expresses "is
this claimable" and whose update claims it, in one atomic Mongo operation —
no separate distributed-lock library or service, since Mongo's per-document
atomicity already proved sufficient for that structurally identical problem
in this same codebase.

Staleness (a pod that crashes mid-debate, so no code ever runs to release
its claim) is handled by an application-level heartbeat comparison baked
into the same atomic filter, not a Mongo TTL index — a TTL index's ~60s
background sweep can't participate in an atomic filter+update, so relying
on it would reintroduce a check-then-act race this design exists to avoid.
The heartbeat itself piggybacks on the orchestrator's existing per-round
checkpoint write; no separate background task.
"""

from __future__ import annotations

import os
import socket
import uuid
from datetime import datetime, timedelta, timezone

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError

# Process-global, generated once at import time — NOT per-request. A fresh
# DebateOrchestrator is constructed per HTTP request (see
# orchestrator_factory.build_orchestrator), so a per-request id would make
# the same pod's own two requests look like different holders and defeat
# the ability to reason about "this pod already has it" vs "a different pod
# has it."
HOLDER_ID: str = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


class RunLockStore:
    def __init__(self, mongo_uri: str, mongo_db: str, lease_seconds: int = 120):
        self._client: AsyncIOMotorClient = AsyncIOMotorClient(mongo_uri)
        self._collection = self._client[mongo_db]["debate_run_locks"]
        self._lease = timedelta(seconds=lease_seconds)
        self._index_ensured = False

    async def ensure_index(self) -> None:
        """Idempotent — safe to call repeatedly with no cross-pod
        coordination. Without this unique index, two concurrent first-ever
        claims for a brand-new run_id could each insert their own document
        instead of racing for the same one. Called lazily on first acquire()
        rather than at construction time: RunLockStore is constructed
        synchronously (orchestrator_factory.build_orchestrator isn't async —
        it's called from 5 places across the CLI and API, some of them
        outside any running event loop at construction time), so this can't
        be awaited or scheduled via asyncio.create_task() there."""
        if self._index_ensured:
            return
        await self._collection.create_index("run_id", unique=True)
        self._index_ensured = True

    async def acquire(self, run_id: str, holder_id: str) -> bool:
        """Atomically claims run_id for holder_id if it's unclaimed,
        explicitly released, or its last heartbeat is older than the lease
        — all three conditions expressed in one filter, evaluated with the
        claiming $set as a single indivisible Mongo operation, mirroring
        BudgetStore.reserve()'s filter+update idiom. Returns True if this
        holder now holds the claim, False if a live holder already does.

        Two distinct races are possible here, both resolved by the unique
        index on run_id, but in different ways that this method has to
        handle differently:
          - Two concurrent first-ever claims for a brand-new run_id (no
            document exists yet): both attempt an upsert-insert; the
            index lets exactly one succeed, and the loser's insert raises
            DuplicateKeyError — caught below, then resolved by re-reading
            who actually won.
          - A claim attempt against a run_id whose document already exists
            but is legitimately held by a live holder (the filter's $or
            correctly excludes it): find_one_and_update still finds "no
            match" and, with upsert=True, still attempts an insert — which
            *also* raises DuplicateKeyError, since a document with that
            run_id already exists. Confirmed live against real MongoDB
            (not caught by this module's original fake-collection-based
            tests, which didn't model pymongo's actual upsert-vs-unique-
            index interaction) — this is the common case in practice, not
            just the rare simultaneous-first-claim race.
        Both cases collapse to the same handling: catch the error, then
        re-read to determine who actually holds the claim now.
        """
        await self.ensure_index()
        now = datetime.now(timezone.utc)
        stale_before = now - self._lease
        try:
            await self._collection.find_one_and_update(
                {
                    "run_id": run_id,
                    "$or": [
                        {"status": "released"},
                        {"heartbeat_at": {"$lt": stale_before}},
                    ],
                },
                {
                    "$set": {
                        "holder_id": holder_id,
                        "claimed_at": now,
                        "heartbeat_at": now,
                        "status": "held",
                    }
                },
                upsert=True,
            )
        except DuplicateKeyError:
            pass
        doc = await self._collection.find_one({"run_id": run_id})
        return doc is not None and doc["holder_id"] == holder_id

    async def heartbeat(self, run_id: str, holder_id: str) -> None:
        """Refreshes heartbeat_at for the current holder only — a no-op,
        not an error, if this holder no longer holds the claim (e.g. its
        lease already expired and someone else reclaimed it); a heartbeat
        failure must never abort an in-progress debate."""
        await self._collection.update_one(
            {"run_id": run_id, "holder_id": holder_id},
            {"$set": {"heartbeat_at": datetime.now(timezone.utc)}},
        )

    async def release(self, run_id: str, holder_id: str) -> None:
        """Marks the claim released, scoped to holder_id so a late release
        from a holder that already lost the claim (its lease expired and
        someone else re-claimed first) cannot clobber the new holder's
        claim."""
        await self._collection.update_one(
            {"run_id": run_id, "holder_id": holder_id},
            {"$set": {"status": "released", "heartbeat_at": datetime.now(timezone.utc)}},
        )

    async def close(self) -> None:
        self._client.close()
