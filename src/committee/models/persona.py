"""Persona-as-data model: the user-facing identity fields that define how an
agent thinks (name, role, responsibility, thinking style, priorities, blind
spots). This is deliberately the *only* thing user input can supply — the
output contract (schema, tool-calling, orchestration/budget behavior) lives
entirely in code (agents/_base_impl.py) and is never touched by this model.

Field length caps exist so this free text can be safely embedded in a
demarcated section of a system prompt (see agents/prompts/persona_template.py)
without ballooning token spend or drowning out the fixed instructions.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

_NAME_MAX = 80
_ROLE_MAX = 120
_RESPONSIBILITY_MAX = 600
_THINKING_STYLE_MAX = 600
_PRIORITY_MAX = 120
_BLIND_SPOT_MAX = 120
_MAX_LIST_ITEMS = 8

# Control characters (other than plain whitespace) have no legitimate use in
# a short identity field and are a common smuggling vector for prompt-format
# delimiters — strip rather than reject, since a user pasting from a rich
# text editor shouldn't get a hard validation failure over an invisible
# character they didn't know was there.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean_text(value: str) -> str:
    value = _CONTROL_CHARS_RE.sub("", value)
    return value.strip()


class PersonaCreate(BaseModel):
    """Inbound shape for POST /agents — exactly the fields Phase 1-A allows a
    user to supply. No field here can influence the output schema or
    orchestration behavior; it only ever becomes prose in a clearly-labeled
    identity section of a system prompt."""

    name: str = Field(min_length=1, max_length=_NAME_MAX)
    role: str = Field(min_length=1, max_length=_ROLE_MAX)
    responsibility: str = Field(min_length=1, max_length=_RESPONSIBILITY_MAX)
    thinking_style: str = Field(min_length=1, max_length=_THINKING_STYLE_MAX)
    priorities: list[str] = Field(min_length=1, max_length=_MAX_LIST_ITEMS)
    blind_spots: list[str] = Field(min_length=1, max_length=_MAX_LIST_ITEMS)

    @field_validator("name", "role", "responsibility", "thinking_style")
    @classmethod
    def _clean_scalar(cls, value: str) -> str:
        cleaned = _clean_text(value)
        if not cleaned:
            raise ValueError("field cannot be blank after sanitization")
        return cleaned

    @field_validator("priorities", "blind_spots")
    @classmethod
    def _clean_list(cls, values: list[str]) -> list[str]:
        cleaned = []
        for item in values:
            item = _clean_text(item)
            if not item:
                continue
            if len(item) > _PRIORITY_MAX:
                item = item[:_PRIORITY_MAX].rstrip()
            cleaned.append(item)
        if not cleaned:
            raise ValueError("at least one non-blank entry is required")
        return cleaned


class AgentPersona(PersonaCreate):
    """The stored record — PersonaCreate's fields plus identity/lifecycle
    metadata that the user never supplies directly."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    is_builtin: bool = False
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PersonaPatch(BaseModel):
    """PATCH /agents/{id} — activation toggle only. Identity fields are
    immutable after creation (mirrors the built-ins, which aren't editable
    either); create a new persona instead of mutating one in place."""

    is_active: bool
