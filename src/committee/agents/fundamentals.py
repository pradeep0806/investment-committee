from committee.agents._base_impl import BaseAnalystAgent
from committee.agents.prompts.fundamentals import SYSTEM_PROMPT
from committee.agents.registry import register


@register("fundamentals")
class FundamentalsAgent(BaseAnalystAgent):
    agent_id = "fundamentals"
    lens_name = "Fundamentals/Valuation"
    system_prompt = SYSTEM_PROMPT
