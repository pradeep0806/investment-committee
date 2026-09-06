from __future__ import annotations

from anthropic import AsyncAnthropic

_TOOL_NAME = "emit_structured_output"


class AnthropicRawCaller:
    def __init__(self, model: str, api_key: str, timeout_seconds: int, temperature: float | None = None):
        self.model = model
        self.temperature = temperature
        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout_seconds)

    async def __call__(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict,
        retry_note: str | None,
        max_tokens: int | None,
    ) -> tuple[dict, int]:
        messages = [{"role": "user", "content": user_prompt}]
        if retry_note:
            messages.append({"role": "user", "content": retry_note})

        extra_kwargs: dict = {}
        if self.temperature is not None:
            extra_kwargs["temperature"] = self.temperature

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens if max_tokens is not None else 4096,
            system=system_prompt,
            messages=messages,
            tools=[
                {
                    "name": _TOOL_NAME,
                    "description": "Emit the structured analysis output.",
                    "input_schema": schema,
                }
            ],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
            **extra_kwargs,
        )

        tokens_used = response.usage.input_tokens + response.usage.output_tokens
        tool_use_block = next((block for block in response.content if block.type == "tool_use"), None)
        if tool_use_block is None:
            # The model can legitimately return no tool_use block at all —
            # e.g. it hit max_tokens mid-thought before emitting one. Treat
            # it as a validation failure the existing retry loop already
            # handles (see litellm_provider.py's identical guard), not a
            # crash — this used to raise an unhandled StopIteration.
            return {}, tokens_used

        return tool_use_block.input, tokens_used
