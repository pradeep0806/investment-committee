"""RedisBus: pub/sub channel per run_id for live streaming (feeds the SSE
endpoint, step 9), plus a hot cache for the agent/prompt registry
(CLAUDE.md §1.3). Deliberately NOT shaped as a TraceStore — it's a different
kind of concern (transient event fan-out + cache, not durable trace
persistence) — so it gets its own narrow interface instead of forcing
save_round/save_final/get_run semantics onto it.

Best-effort like Mongo: publish failures are caught and logged, never
allowed to block the debate.
"""

from __future__ import annotations

import json

import redis.asyncio as redis
import structlog

logger = structlog.get_logger()

_CHANNEL_PREFIX = "committee:debate:"


class RedisBus:
    def __init__(self, redis_url: str):
        self._client = redis.from_url(redis_url)

    @staticmethod
    def _channel_for(run_id: str) -> str:
        return f"{_CHANNEL_PREFIX}{run_id}"

    async def publish_round_event(self, run_id: str, event: dict) -> None:
        try:
            await self._client.publish(self._channel_for(run_id), json.dumps(event))
        except Exception as exc:
            logger.warning(
                "redis_publish_failed", run_id=run_id, event_type=event.get("event"), error=str(exc)
            )

    async def subscribe(self, run_id: str):
        pubsub = self._client.pubsub()
        await pubsub.subscribe(self._channel_for(run_id))
        return pubsub

    async def get_cached_registry(self, key: str) -> str | None:
        try:
            value = await self._client.get(key)
            return value.decode("utf-8") if value is not None else None
        except Exception as exc:
            logger.warning("redis_get_failed", key=key, error=str(exc))
            return None

    async def set_cached_registry(self, key: str, value: str, ttl_seconds: int = 3600) -> None:
        try:
            await self._client.set(key, value, ex=ttl_seconds)
        except Exception as exc:
            logger.warning("redis_set_failed", key=key, error=str(exc))

    async def close(self) -> None:
        await self._client.aclose()
