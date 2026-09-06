from committee.agents._base_impl import BaseAnalystAgent
from committee.agents.prompts.market_sentiment import SYSTEM_PROMPT
from committee.agents.registry import register


@register("market_sentiment")
class MarketSentimentAgent(BaseAnalystAgent):
    agent_id = "market_sentiment"
    lens_name = "Market Sentiment"
    system_prompt = SYSTEM_PROMPT
