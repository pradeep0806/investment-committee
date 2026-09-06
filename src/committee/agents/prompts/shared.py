"""Guardrails shared by every agent's prompt, so the "independent reasoning in
round 1, rebuttal-aware in later rounds" rule (CLAUDE.md §5 step 1) lives in one
place instead of being copy-pasted into each lens's prompt file."""

from __future__ import annotations

from committee.models.agent_output import AgentOutput
from committee.models.requests import ThesisRequest

ROUND_ONE_GUARDRAIL = (
    "This is round 1 of the debate. Give your own independent view based solely on your "
    "lens — do not try to agree with anyone, since no one else has spoken yet."
)

LATER_ROUND_GUARDRAIL = (
    "You may update your position in light of the other analysts' prior-round arguments "
    "below, but only if their reasoning genuinely changes your assessment through your "
    "lens — do not converge just to agree. State clearly if and why you're standing firm "
    "or shifting."
)


def render_prior_outputs(prior_round_outputs: list[AgentOutput] | None) -> str:
    if not prior_round_outputs:
        return ""
    lines = ["Prior round arguments from the other analysts:"]
    for output in prior_round_outputs:
        lines.append(
            f"- [{output.agent_id}] stance={output.stance.value} "
            f"confidence={output.confidence} key_factors={output.key_factors} "
            f"top_risk={output.top_risk!r}"
        )
    return "\n".join(lines)


def build_user_prompt(
    request: ThesisRequest,
    round: int,
    prior_round_outputs: list[AgentOutput] | None,
    directive: str | None,
) -> str:
    parts = [f"Investment thesis under review: {request.thesis}"]
    if request.entity:
        parts.append(f"Entity/ticker context: {request.entity}")
    if request.priors:
        parts.append(f"Known priors: {request.priors}")

    parts.append(ROUND_ONE_GUARDRAIL if round == 1 else LATER_ROUND_GUARDRAIL)

    prior_text = render_prior_outputs(prior_round_outputs)
    if prior_text:
        parts.append(prior_text)

    if directive:
        parts.append(f"Special instruction for this round: {directive}")

    parts.append(
        "Respond only through the provided tool, with a stance (Buy/Hold/Sell/Pass), a "
        "confidence 0-100, 2-5 short key_factors tags, and your single top_risk."
    )
    return "\n\n".join(parts)
