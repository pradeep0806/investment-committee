from committee.agents._base_impl import BaseAnalystAgent
from committee.agents.prompts.macro_context import SYSTEM_PROMPT
from committee.agents.registry import register


@register("macro_context")
class MacroContextAgent(BaseAnalystAgent):
    agent_id = "macro_context"
    lens_name = "Macro/Industry Context"
    system_prompt = SYSTEM_PROMPT
