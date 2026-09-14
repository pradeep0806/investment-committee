"""DynamicAnalystAgent: an AnalystAgent built at debate-start time from a
stored AgentPersona record, rather than from a hand-written subclass.

This is the "persona as data" half of the seam described in CLAUDE.md's
extension: the *contract* (BaseAnalystAgent.analyze, the tool schema, budget
gating) is identical to every built-in agent and lives entirely in
_base_impl.py, untouched by this class. The only thing this class does is
translate a persona record's free-text identity fields into the system
prompt via render_persona_system_prompt, which is the one function allowed
to embed untrusted text into an LLM-bound string (see prompts/
persona_template.py for the guardrails).

Deliberately not decorated with @register / not listed in AGENT_MODULES —
built-ins are registered at import time because there's a fixed, known set of
them; custom personas are looked up from Mongo per debate, so they're
constructed directly by registry.build_agents() instead (see registry.py).
"""

from __future__ import annotations

from committee.agents._base_impl import BaseAnalystAgent
from committee.agents.prompts.persona_template import render_persona_system_prompt
from committee.models.persona import AgentPersona
from committee.orchestration.budget_gate import BudgetGate


class DynamicAnalystAgent(BaseAnalystAgent):
    def __init__(self, budget_gate: BudgetGate, persona: AgentPersona):
        super().__init__(
            budget_gate=budget_gate,
            agent_id=persona.id,
            lens_name=persona.name,
            system_prompt=render_persona_system_prompt(persona),
        )
        self.persona = persona
