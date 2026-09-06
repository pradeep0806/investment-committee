from __future__ import annotations

import json

from openai import AsyncOpenAI

_TOOL_NAME = "emit_structured_output"


class OpenAIRawCaller:
    def __init__(self, model: str, api_key: str, timeout_seconds: int, temperature: float | None = None):
        self.model = model
        self.temperature = temperature
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout_seconds)

    async def __call__(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict,
        retry_note: str | None,
        max_tokens: int | None,
    ) -> tuple[dict, int]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if retry_note:
            messages.append({"role": "user", "content": retry_note})

        extra_kwargs: dict = {}
        if max_tokens is not None:
            extra_kwargs["max_completion_tokens"] = max_tokens
        if self.temperature is not None:
            extra_kwargs["temperature"] = self.temperature

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": _TOOL_NAME,
                        "description": "Emit the structured analysis output.",
                        "parameters": schema,
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": _TOOL_NAME}},
            **extra_kwargs,
        )

        tokens_used = response.usage.total_tokens if response.usage else 0
        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            # See litellm_provider.py's identical guard: the model can
            # legitimately return no tool call (e.g. hit max_tokens before
            # emitting one) — treat it as a validation failure the existing
            # retry loop already handles, not a crash.
            return {}, tokens_used

        raw_args = json.loads(tool_calls[0].function.arguments)
        return raw_args, tokens_used
