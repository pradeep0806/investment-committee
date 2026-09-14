"""PersonaStore: Mongo-backed CRUD for user-defined and built-in analyst
personas (the `agent_personas` collection).

Built-ins are seeded rows, not special-cased code (per CLAUDE.md's "no code
change to add a 5th agent" spirit, extended here to "no code change to see
the existing four as personas too") — `ensure_builtins_seeded` upserts one
document per standing agent using the exact same PersonaCreate shape a user's
POST /agents would use. The four standing agent *classes*
(agents/fundamentals.py etc.) are unaffected by this and keep working exactly
as before; the seeded rows exist purely so GET /agents can list the full
roster (built-in + custom) from one place, and so the debate request's
`agent_ids` selection can reference built-ins by the same id space as custom
personas.
"""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient

from committee.models.persona import AgentPersona, PersonaCreate

# Mirrors of the four standing agents' SYSTEM_PROMPT files, decomposed into
# persona fields. These do NOT drive the standing agents' actual behavior —
# fundamentals.py/market_sentiment.py/risk_contrarian.py/macro_context.py
# still use their own SYSTEM_PROMPT constants unchanged. This mapping exists
# solely so the seeded rows describe the real built-ins accurately for
# listing/selection purposes.
_BUILTIN_PERSONAS: dict[str, PersonaCreate] = {
    "fundamentals": PersonaCreate(
        name="Fundamentals/Valuation",
        role="Fundamentals/Valuation analyst",
        responsibility=(
            "Assess financials, unit economics, valuation multiples, revenue quality, "
            "margins, balance sheet strength, and whether the price paid is justified by "
            "the underlying business."
        ),
        thinking_style="Numbers-first; discounts narrative and momentum entirely.",
        priorities=["Valuation discipline", "Balance sheet strength", "Earnings quality"],
        blind_spots=["Narrative and price momentum carry zero weight in this lens"],
    ),
    "market_sentiment": PersonaCreate(
        name="Market Sentiment",
        role="Market Sentiment analyst",
        responsibility=(
            "Assess price action, market positioning, narrative momentum, how the market "
            "is pricing the story relative to consensus, and what catalysts could shift "
            "sentiment."
        ),
        thinking_style="Reads positioning and narrative, not financial statements.",
        priorities=["Price action", "Narrative momentum", "Catalyst timing"],
        blind_spots=["Weak on balance-sheet detail and accounting quality"],
    ),
    "risk_contrarian": PersonaCreate(
        name="Risk Contrarian",
        role="Risk Contrarian analyst",
        responsibility=(
            "Identify downside scenarios, tail risk, what could go wrong, management "
            "credibility issues, and the ways a consensus thesis quietly fails."
        ),
        thinking_style="Actively tries to break the thesis before conceding it might hold.",
        priorities=["Tail risk", "Thesis stress-testing", "Management credibility"],
        blind_spots=["Deliberately skeptical even of strong theses, by design"],
    ),
    "macro_context": PersonaCreate(
        name="Macro/Industry Context",
        role="Macro/Industry Context analyst",
        responsibility=(
            "Assess sector-wide trends, competitive dynamics, and macro headwinds/"
            "tailwinds, and how the company sits relative to its industry's trajectory."
        ),
        thinking_style="Evaluates whether the wind favors the company, not company execution.",
        priorities=["Sector trends", "Competitive dynamics", "Macro cycle positioning"],
        blind_spots=["Weak on company-specific execution detail"],
    ),
}


class PersonaStore:
    def __init__(self, mongo_uri: str, mongo_db: str):
        self._client: AsyncIOMotorClient = AsyncIOMotorClient(mongo_uri)
        self._collection = self._client[mongo_db]["agent_personas"]

    async def ensure_builtins_seeded(self) -> None:
        """Upserts the four built-in personas with fixed, well-known ids
        (their existing agent_id strings) so they're stable across restarts
        and match the registry's DEFAULT_AGENT_ROLES ids exactly."""
        for agent_id, persona_create in _BUILTIN_PERSONAS.items():
            persona = AgentPersona(id=agent_id, is_builtin=True, **persona_create.model_dump())
            await self._collection.update_one(
                {"_id": agent_id},
                {"$setOnInsert": persona.model_dump(mode="json")},
                upsert=True,
            )

    async def create(self, persona_create: PersonaCreate) -> AgentPersona:
        persona = AgentPersona(is_builtin=False, **persona_create.model_dump())
        document = persona.model_dump(mode="json")
        document["_id"] = persona.id
        await self._collection.insert_one(document)
        return persona

    async def list_all(self) -> list[AgentPersona]:
        documents = await self._collection.find({}).to_list(length=None)
        return [_from_document(doc) for doc in documents]

    async def get(self, persona_id: str) -> AgentPersona | None:
        document = await self._collection.find_one({"_id": persona_id})
        return _from_document(document) if document else None

    async def set_active(self, persona_id: str, is_active: bool) -> AgentPersona | None:
        document = await self._collection.find_one_and_update(
            {"_id": persona_id},
            {"$set": {"is_active": is_active}},
            return_document=True,
        )
        return _from_document(document) if document else None

    async def list_active_custom(self) -> list[AgentPersona]:
        documents = await self._collection.find({"is_builtin": False, "is_active": True}).to_list(
            length=None
        )
        return [_from_document(doc) for doc in documents]

    async def close(self) -> None:
        self._client.close()


def _from_document(document: dict) -> AgentPersona:
    document = dict(document)
    document.pop("_id", None)
    return AgentPersona.model_validate(document)
