from committee.agents._base_impl import BaseAnalystAgent
from committee.agents.prompts.risk_contrarian import SYSTEM_PROMPT
from committee.agents.registry import register


@register("risk_contrarian")
class RiskContrarianAgent(BaseAnalystAgent):
    agent_id = "risk_contrarian"
    lens_name = "Risk Contrarian"
    system_prompt = SYSTEM_PROMPT
