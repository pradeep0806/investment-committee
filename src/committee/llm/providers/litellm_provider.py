from __future__ import annotations

import json

import litellm

_TOOL_NAME = "emit_structured_output"


class LiteLLMRawCaller:
    """Routes through LiteLLM's unified interface — covers any provider not given
    its own dedicated wrapper (Anthropic and OpenAI have native ones above for
    the two most common cases; everything else goes through here via LLM_MODEL,
    e.g. "gemini/gemini-1.5-pro" for plain API-key auth, "vertex_ai/gemini-1.5-pro"
    for GCP service-account auth via GOOGLE_APPLICATION_CREDENTIALS, or
    "ollama/<model>" for a local model server (no API key needed at all —
    Ollama has no auth by default; `base_url` points at wherever it's running,
    typically http://localhost:11434). Confirmed via litellm.supports_function_calling
    that Ollama models support the same tool-calling path everything else here
    relies on for structured output — this isn't a degraded/best-effort mode.

    Vertex AI auth does not flow through `api_key` at all — LiteLLM's vertex_ai
    path authenticates via Application Default Credentials, resolved from the
    GOOGLE_APPLICATION_CREDENTIALS env var (a path to a service-account JSON key)
    plus an explicit GCP project/location, both read from Settings and passed
    through here rather than folded into the generic `api_key` field.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        timeout_seconds: int,
        vertex_project: str | None = None,
        vertex_location: str | None = None,
        temperature: float | None = None,
        thinking_budget: int | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.vertex_project = vertex_project
        self.vertex_location = vertex_location
        self.temperature = temperature
        # Explicit override, when given, wins over the max_tokens // 2
        # default derived below — lets a caller (e.g. the frontend) set
        # Gemini's thinking budget directly rather than only indirectly via
        # max_tokens.
        self.thinking_budget = thinking_budget
        # Only meaningful for a local/self-hosted model server (Ollama) —
        # hosted providers resolve their own endpoint and ignore this.
        self.base_url = base_url

    def _is_gemini_model(self) -> bool:
        return "gemini" in self.model.lower()

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
        call_model = self.model
        if self.model.startswith("vertex_ai/"):
            # Service-account auth: GOOGLE_APPLICATION_CREDENTIALS (set in the
            # process environment) supplies the credentials themselves; project
            # and location are required alongside it and are not inferred.
            extra_kwargs["vertex_project"] = self.vertex_project
            extra_kwargs["vertex_location"] = self.vertex_location
        elif self.model.startswith("ollama/") or self.model.startswith("ollama_chat/"):
            # litellm's plain "ollama/" provider talks to Ollama's legacy
            # /api/generate endpoint and fakes tool-calling by stuffing the
            # tool schema into the prompt text and asking the model to
            # emit raw JSON back — no real tool-calling protocol involved.
            # In practice this made every structured-output call fail
            # validation (confirmed live: qwen3:14b and others returned an
            # empty "{}" after ~2 tokens of generation, every single time,
            # despite litellm.supports_function_calling() reporting True).
            # "ollama_chat/" instead routes through Ollama's native
            # /api/chat endpoint, which has real tool-calling support — the
            # same request against ollama_chat/qwen3:14b returns a proper
            # tool_calls entry on the first try. Rewriting the prefix here
            # (rather than requiring "ollama_chat/" in config/the frontend)
            # keeps "ollama/<model>" working as the one documented format.
            call_model = "ollama_chat/" + self.model.split("/", 1)[1]
            # No API key — Ollama has no auth by default. api_base tells
            # litellm where the local server actually is; litellm's own
            # default already matches Ollama's (http://localhost:11434), but
            # passing it explicitly makes a non-default host/port possible.
            extra_kwargs["api_base"] = self.base_url or "http://localhost:11434"
            # Force a fresh connection per call rather than reusing litellm's
            # pooled keep-alive one. This matters specifically when base_url
            # is host.docker.internal (running the API in Docker against a
            # host-side Ollama): Docker Desktop's NAT can silently drop an
            # idle container->host mapping without either side seeing a
            # close, so httpx reuses what looks like a live pooled
            # connection but is actually dead — the read then hangs forever
            # (observed: a debate went silent for minutes with no timeout
            # or error, while a brand-new connection to the same host:port
            # succeeded instantly). Connection: close sidesteps this by
            # never reusing a pooled socket for Ollama traffic.
            extra_kwargs["extra_headers"] = {"Connection": "close"}
        else:
            extra_kwargs["api_key"] = self.api_key

        if self.temperature is not None:
            extra_kwargs["temperature"] = self.temperature

        if max_tokens is not None:
            extra_kwargs["max_tokens"] = max_tokens
            if self._is_gemini_model():
                # Gemini's "thinking" tokens are a distinct budget dimension
                # from output tokens (Gemini 2.5+ has thinking enabled by
                # default) — capping max_tokens alone leaves thinking spend
                # unbounded, which is exactly what was blowing past
                # token_budget in practice. litellm's `thinking` param
                # translates to Gemini's `thinkingBudget` field (see
                # VertexGeminiConfig._map_thinking_param). An explicit
                # thinking_budget override wins if given; otherwise split
                # the cap roughly in half between thinking and output — a
                # deliberate simplification over something more precisely
                # tuned, since neither this project nor litellm exposes a
                # "total including thinking" single knob for Gemini.
                thinking_budget = (
                    self.thinking_budget if self.thinking_budget is not None else max(max_tokens // 2, 128)
                )
                extra_kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

        response = await litellm.acompletion(
            model=call_model,
            timeout=self.timeout_seconds,
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
            # The model can legitimately return no tool call at all — e.g.
            # it hit max_tokens mid-thought before emitting one, or (with a
            # very tight thinking budget) spent its whole allocation
            # reasoning and had nothing left for the actual call. Returning
            # {} rather than raising lets Pydantic's own validation reject
            # it as a missing-fields error, which call_structured's existing
            # retry loop already knows how to handle (feed the error back,
            # try again) — this is an ordinary structured-output failure,
            # not a crash-worthy one.
            return {}, tokens_used

        raw_args = json.loads(tool_calls[0].function.arguments)
        return raw_args, tokens_used
