"""MongoStore: motor-based TraceStore implementation. Never used directly by
the orchestrator — always wrapped in BestEffortTraceStore, since Mongo
writes are best-effort and must never block or crash the debate (CLAUDE.md
§1.3, §5 step 7).
"""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient

from committee.models.trace import DebateTrace, RoundRecord


class MongoStore:
    def __init__(self, mongo_uri: str, mongo_db: str):
        self._client: AsyncIOMotorClient = AsyncIOMotorClient(mongo_uri)
        self._collection = self._client[mongo_db]["debate_traces"]

    async def save_round(self, run_id: str, round_record: RoundRecord) -> None:
        await self._collection.update_one(
            {"run_id": run_id},
            {"$push": {"rounds": round_record.model_dump(mode="json")}},
            upsert=True,
        )

    async def save_final(self, run_id: str, trace: DebateTrace) -> None:
        await self._collection.replace_one(
            {"run_id": run_id}, trace.model_dump(mode="json"), upsert=True
        )

    async def get_run(self, run_id: str) -> DebateTrace | None:
        document = await self._collection.find_one({"run_id": run_id})
        if document is None:
            return None
        document.pop("_id", None)
        return DebateTrace.model_validate(document)

    async def list_runs(self) -> list[str]:
        run_ids = await self._collection.distinct("run_id")
        return sorted(run_ids)

    async def close(self) -> None:
        self._client.close()
