"""API request/response schemas. Thin wrappers/aliases over the domain
models — ThesisRequest/DebateConfig already have exactly the shape a client
needs to send, and DebateTrace already has exactly the shape a client needs
back, so there's no separate API-only representation to maintain in sync.
"""

from __future__ import annotations

from pydantic import BaseModel

from committee.models.persona import AgentPersona, PersonaCreate, PersonaPatch
from committee.models.requests import DebateConfig, ThesisRequest


class DebateRequestBody(BaseModel):
    request: ThesisRequest
    config: DebateConfig = DebateConfig()


class HealthResponse(BaseModel):
    status: str = "ok"


# PersonaCreate/AgentPersona/PersonaPatch (models/persona.py) already have
# exactly the shape the /agents endpoints need to send and receive — same
# "no separate API-only representation" reasoning as DebateRequestBody above.
AgentCreateRequest = PersonaCreate
AgentResponse = AgentPersona
AgentPatchRequest = PersonaPatch


class AgentListResponse(BaseModel):
    agents: list[AgentPersona]
