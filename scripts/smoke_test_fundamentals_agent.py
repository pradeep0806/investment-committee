"""Manual smoke test: one real LLM call through FundamentalsAgent.

Not a pytest test (no mocking) — run this by hand once `.env` has real
credentials, to confirm the provider wrapper genuinely works end-to-end
against a live API. For Vertex AI service-account auth specifically:

    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
    # .env: LLM_PROVIDER=litellm
    #       LLM_MODEL=vertex_ai/gemini-2.0-flash
    #       LLM_VERTEX_PROJECT=<your-gcp-project-id>
    #       LLM_VERTEX_LOCATION=us-central1

Usage:
    source .venv/bin/activate
    python scripts/smoke_test_fundamentals_agent.py
"""

import asyncio

from committee.agents.fundamentals import FundamentalsAgent
from committee.config import get_settings
from committee.llm.client import build_llm_client_from_settings
from committee.models.requests import ThesisRequest


async def main() -> None:
    settings = get_settings()
    print(f"provider={settings.llm_provider} model={settings.llm_model}")

    client = build_llm_client_from_settings(settings)
    agent = FundamentalsAgent(llm_client=client)

    output = await agent.analyze(
        request=ThesisRequest(
            thesis=(
                "NovaTech Inc. is showing 40% YoY revenue growth with expanding enterprise "
                "customer count and a newly hired AI infrastructure CTO."
            ),
            entity="NovaTech Inc.",
        ),
        round=1,
        token_budget=2000,
        prior_round_outputs=None,
    )

    print(output.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
