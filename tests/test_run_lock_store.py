"""Tests RunLockStore's atomic-claim contract against a fake Mongo
collection that faithfully implements find_one_and_update's filter+update
semantics, including upsert (no real Mongo in this test environment —
matches the existing pattern, see test_budget_store.py). The point under
test is the same atomicity property BudgetStore relies on: a filtered
find_one_and_update is a single indivisible step, so two coroutines racing
to claim the same run_id cannot both succeed.

The fake's find_one_and_update deliberately mirrors real pymongo's actual
(surprising) behavior, confirmed live against a real MongoDB container: when
upsert=True and the filter finds no matching document, pymongo attempts an
INSERT regardless of *why* nothing matched — including when a document with
that run_id already exists but is legitimately excluded by the filter (e.g.
held by a live holder). That insert collides with the unique index on
run_id and raises DuplicateKeyError, not a quiet no-op. A first version of
this fake got this wrong (silently no-op'd instead), which meant these
tests never actually exercised RunLockStore.acquire()'s DuplicateKeyError
handling — caught only by testing against real Mongo.
"""

import asyncio
from datetime import datetime, timedelta, timezone

from pymongo.errors import DuplicateKeyError

from committee.storage.run_lock_store import RunLockStore


class _FakeLockCollection:
    """Single-document-per-run_id fake supporting the filter shapes
    RunLockStore.acquire()/heartbeat()/release() actually use: an $or of
    {"status": "released"} / {"heartbeat_at": {"$lt": ...}}, a $set update,
    and upsert=True — including raising DuplicateKeyError on an upsert
    attempt against a run_id that already has a (non-matching) document, to
    faithfully reproduce real pymongo's behavior. No await between the
    filter check and the mutation — the same no-interleaving property that
    makes this a faithful atomicity stand-in for real Mongo's per-document
    guarantee."""

    def __init__(self):
        self._docs: dict[str, dict] = {}

    def _matches(self, doc: dict | None, filter_: dict) -> bool:
        if doc is None:
            return False
        for or_clause in filter_.get("$or", []):
            for field, condition in or_clause.items():
                value = doc.get(field)
                if isinstance(condition, dict) and "$lt" in condition:
                    if value is not None and value < condition["$lt"]:
                        return True
                elif value == condition:
                    return True
        return False

    async def find_one_and_update(self, filter_, update, upsert=False):
        run_id = filter_["run_id"]
        doc = self._docs.get(run_id)
        if doc is None:
            if upsert:
                self._docs[run_id] = dict(update.get("$set", {}))
                self._docs[run_id]["run_id"] = run_id
            return None
        if self._matches(doc, filter_):
            doc.update(update.get("$set", {}))
            return doc
        if upsert:
            # Real pymongo: "no document matched" + upsert=True always means
            # "attempt an insert," even when a document with this key exists
            # but was excluded by the filter — the insert then collides with
            # the unique index. Not a quiet no-op.
            raise DuplicateKeyError(f"E11000 duplicate key error: run_id={run_id!r}")
        return None

    async def update_one(self, filter_, update):
        run_id = filter_["run_id"]
        doc = self._docs.get(run_id)
        if doc is None:
            return
        if "holder_id" in filter_ and doc.get("holder_id") != filter_["holder_id"]:
            return  # scoped update targeting a holder that no longer matches
        doc.update(update.get("$set", {}))

    async def find_one(self, filter_):
        return self._docs.get(filter_["run_id"])


def _make_store(lease_seconds: int = 120) -> RunLockStore:
    store = RunLockStore.__new__(RunLockStore)
    store._collection = _FakeLockCollection()
    store._lease = timedelta(seconds=lease_seconds)
    store._index_ensured = True  # skip create_index against the fake
    return store


class TestRunLockStoreAtomicity:
    async def test_acquire_on_unclaimed_run_id_succeeds(self):
        store = _make_store()
        assert await store.acquire("run-1", "holder-a") is True

    async def test_acquire_when_already_held_by_a_live_holder_fails(self):
        store = _make_store()
        assert await store.acquire("run-1", "holder-a") is True
        assert await store.acquire("run-1", "holder-b") is False
        doc = await store._collection.find_one({"run_id": "run-1"})
        assert doc["holder_id"] == "holder-a"  # untouched by the losing attempt

    async def test_acquire_after_release_by_original_holder_succeeds(self):
        store = _make_store()
        await store.acquire("run-1", "holder-a")
        await store.release("run-1", "holder-a")
        assert await store.acquire("run-1", "holder-b") is True

    async def test_acquire_with_expired_heartbeat_is_claimable_by_a_new_holder(self):
        """The crash-recovery case: a holder that stopped heartbeating
        (crashed, OOM-killed) must not wedge the run_id forever."""
        store = _make_store(lease_seconds=60)
        await store.acquire("run-1", "holder-a")
        # Simulate a stale heartbeat by backdating it directly.
        doc = await store._collection.find_one({"run_id": "run-1"})
        doc["heartbeat_at"] = datetime.now(timezone.utc) - timedelta(seconds=61)

        assert await store.acquire("run-1", "holder-b") is True
        doc = await store._collection.find_one({"run_id": "run-1"})
        assert doc["holder_id"] == "holder-b"

    async def test_acquire_with_fresh_heartbeat_is_not_claimable_even_if_status_is_held(self):
        store = _make_store(lease_seconds=60)
        await store.acquire("run-1", "holder-a")
        doc = await store._collection.find_one({"run_id": "run-1"})
        doc["heartbeat_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)

        assert await store.acquire("run-1", "holder-b") is False

    async def test_release_by_a_holder_that_no_longer_holds_the_lock_does_not_clobber_the_new_holder(
        self,
    ):
        store = _make_store(lease_seconds=60)
        await store.acquire("run-1", "holder-a")
        doc = await store._collection.find_one({"run_id": "run-1"})
        doc["heartbeat_at"] = datetime.now(timezone.utc) - timedelta(seconds=61)
        await store.acquire("run-1", "holder-b")  # holder-b now owns it

        await store.release("run-1", "holder-a")  # late release from the old holder

        doc = await store._collection.find_one({"run_id": "run-1"})
        assert doc["holder_id"] == "holder-b"
        assert doc["status"] == "held"

    async def test_concurrent_acquire_attempts_for_the_same_new_run_id_only_one_succeeds(self):
        store = _make_store()

        results = await asyncio.gather(
            store.acquire("run-1", "holder-a"),
            store.acquire("run-1", "holder-b"),
        )

        assert sorted(results) == [False, True]

    async def test_heartbeat_refreshes_heartbeat_at_for_the_current_holder_only(self):
        store = _make_store()
        await store.acquire("run-1", "holder-a")
        doc = await store._collection.find_one({"run_id": "run-1"})
        original_heartbeat = doc["heartbeat_at"]
        doc["heartbeat_at"] = original_heartbeat - timedelta(seconds=5)

        await store.heartbeat("run-1", "holder-b")  # not the current holder
        doc = await store._collection.find_one({"run_id": "run-1"})
        assert doc["heartbeat_at"] == original_heartbeat - timedelta(seconds=5)

        await store.heartbeat("run-1", "holder-a")
        doc = await store._collection.find_one({"run_id": "run-1"})
        assert doc["heartbeat_at"] > original_heartbeat - timedelta(seconds=5)

    async def test_acquire_against_a_live_held_run_id_does_not_raise_duplicate_key_error(self):
        """Real bug found live against actual MongoDB, not caught by an
        earlier version of this test file's fake collection: a claim
        attempt against a run_id whose document already exists (held by a
        live holder) triggers pymongo's upsert=True insert path, which
        collides with the unique index on run_id and raises
        DuplicateKeyError rather than cleanly returning False. acquire()
        must catch this and return False, not let the exception propagate
        and crash the caller (which, in production, would have crashed the
        orchestrator's in-flight debate entirely on any concurrent resume
        attempt — far worse than the rejection this is meant to produce)."""
        store = _make_store()
        await store.acquire("run-1", "holder-a")
        # Must not raise, and must correctly report the claim as refused.
        assert await store.acquire("run-1", "holder-b") is False
