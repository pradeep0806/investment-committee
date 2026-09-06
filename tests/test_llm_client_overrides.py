"""Tests for the per-debate LLM overrides: temperature, thinking_budget,
provider/model switching (including Ollama), all threaded through
LLMClient/build_llm_client_from_settings without touching .env."""

from committee.config import Settings
from committee.llm.client import build_llm_client_from_settings
from committee.llm.providers.litellm_provider import LiteLLMRawCaller


def _settings(**overrides) -> Settings:
    return Settings(llm_api_key="fake-key", **overrides)


def test_build_llm_client_uses_settings_defaults_when_no_overrides_given():
    settings = _settings(llm_provider="anthropic", llm_model="claude-sonnet-4-6")
    client = build_llm_client_from_settings(settings)
    assert client.provider == "anthropic"
    assert client.model == "claude-sonnet-4-6"
    assert client.temperature is None
    assert client.thinking_budget is None


def test_build_llm_client_provider_and_model_override_settings():
    settings = _settings(llm_provider="anthropic", llm_model="claude-sonnet-4-6")
    client = build_llm_client_from_settings(settings, provider="litellm", model="ollama/llama3.1")
    assert client.provider == "litellm"
    assert client.model == "ollama/llama3.1"


def test_build_llm_client_temperature_and_thinking_budget_pass_through():
    settings = _settings()
    client = build_llm_client_from_settings(settings, temperature=0.9, thinking_budget=512)
    assert client.temperature == 0.9
    assert client.thinking_budget == 512


def test_build_llm_client_resolves_ollama_base_url_from_settings():
    settings = _settings(ollama_base_url="http://my-ollama-host:11434")
    client = build_llm_client_from_settings(settings, provider="litellm", model="ollama/llama3.1")
    assert client.base_url == "http://my-ollama-host:11434"


def test_build_llm_client_does_not_set_base_url_for_non_ollama_models():
    settings = _settings(ollama_base_url="http://my-ollama-host:11434")
    client = build_llm_client_from_settings(settings, provider="litellm", model="vertex_ai/gemini-2.5-flash")
    assert client.base_url is None


async def test_litellm_raw_caller_passes_temperature_to_acompletion(monkeypatch):
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop before a real network call")

    import committee.llm.providers.litellm_provider as provider_module

    monkeypatch.setattr(provider_module.litellm, "acompletion", fake_acompletion)

    caller = LiteLLMRawCaller(
        model="gemini/gemini-2.5-flash",
        api_key="fake-key",
        timeout_seconds=60,
        temperature=0.3,
    )
    try:
        await caller(system_prompt="sys", user_prompt="user", schema={}, retry_note=None, max_tokens=500)
    except RuntimeError:
        pass

    assert captured["temperature"] == 0.3


async def test_litellm_raw_caller_explicit_thinking_budget_overrides_default_derivation(monkeypatch):
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop before a real network call")

    import committee.llm.providers.litellm_provider as provider_module

    monkeypatch.setattr(provider_module.litellm, "acompletion", fake_acompletion)

    caller = LiteLLMRawCaller(
        model="vertex_ai/gemini-2.5-flash",
        api_key="fake-key",
        timeout_seconds=60,
        thinking_budget=999,
    )
    try:
        await caller(system_prompt="sys", user_prompt="user", schema={}, retry_note=None, max_tokens=500)
    except RuntimeError:
        pass

    assert captured["thinking"] == {"type": "enabled", "budget_tokens": 999}


async def test_litellm_raw_caller_derives_default_thinking_budget_when_not_overridden(monkeypatch):
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop before a real network call")

    import committee.llm.providers.litellm_provider as provider_module

    monkeypatch.setattr(provider_module.litellm, "acompletion", fake_acompletion)

    caller = LiteLLMRawCaller(model="vertex_ai/gemini-2.5-flash", api_key="fake-key", timeout_seconds=60)
    try:
        await caller(system_prompt="sys", user_prompt="user", schema={}, retry_note=None, max_tokens=500)
    except RuntimeError:
        pass

    assert captured["thinking"] == {"type": "enabled", "budget_tokens": 250}


async def test_litellm_raw_caller_uses_api_base_and_no_api_key_for_ollama(monkeypatch):
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop before a real network call")

    import committee.llm.providers.litellm_provider as provider_module

    monkeypatch.setattr(provider_module.litellm, "acompletion", fake_acompletion)

    caller = LiteLLMRawCaller(
        model="ollama/llama3.1",
        api_key="",
        timeout_seconds=60,
        base_url="http://localhost:11434",
    )
    try:
        await caller(system_prompt="sys", user_prompt="user", schema={}, retry_note=None, max_tokens=500)
    except RuntimeError:
        pass

    assert captured["api_base"] == "http://localhost:11434"
    assert "api_key" not in captured


async def test_litellm_raw_caller_rewrites_ollama_prefix_to_ollama_chat(monkeypatch):
    """Real bug found via live testing: litellm's plain "ollama/" provider
    talks to Ollama's legacy /api/generate endpoint and fakes tool-calling
    by stuffing the schema into the prompt as text, asking the model to
    reply with raw JSON — no real tool-calling protocol. This reliably
    failed structured-output validation against multiple real local models
    (confirmed: qwen3:14b, which litellm.supports_function_calling reports
    as tool-capable, returned an empty "{}" after ~2 tokens, every time).
    "ollama_chat/<model>" instead routes through Ollama's native /api/chat
    endpoint, which has real tool-calling support and returns a proper
    tool_calls entry — verified live against the same model. The caller
    still accepts "ollama/<model>" as the one documented config format and
    rewrites it internally, so this fix needs no frontend/config change."""
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop before a real network call")

    import committee.llm.providers.litellm_provider as provider_module

    monkeypatch.setattr(provider_module.litellm, "acompletion", fake_acompletion)

    caller = LiteLLMRawCaller(model="ollama/qwen3:14b", api_key="", timeout_seconds=60)
    try:
        await caller(system_prompt="sys", user_prompt="user", schema={}, retry_note=None, max_tokens=500)
    except RuntimeError:
        pass

    assert captured["model"] == "ollama_chat/qwen3:14b"


async def test_litellm_raw_caller_forces_connection_close_for_ollama(monkeypatch):
    """Ollama traffic must never reuse a pooled keep-alive connection: a
    Docker container talking to host.docker.internal can have its NAT
    mapping silently dropped by Docker Desktop while idle, leaving httpx's
    pooled connection looking alive but actually dead — the next call then
    hangs forever on read with no timeout or error surfaced. Connection:
    close forces a fresh connection every call, sidestepping this."""
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop before a real network call")

    import committee.llm.providers.litellm_provider as provider_module

    monkeypatch.setattr(provider_module.litellm, "acompletion", fake_acompletion)

    caller = LiteLLMRawCaller(
        model="ollama/llama3.1",
        api_key="",
        timeout_seconds=60,
        base_url="http://host.docker.internal:11434",
    )
    try:
        await caller(system_prompt="sys", user_prompt="user", schema={}, retry_note=None, max_tokens=500)
    except RuntimeError:
        pass

    assert captured["extra_headers"] == {"Connection": "close"}


async def test_litellm_raw_caller_returns_empty_dict_when_no_tool_call_in_response(monkeypatch):
    """Real bug found via live testing: the model can legitimately return no
    tool call at all (e.g. hit max_tokens mid-thought, or spent a very tight
    thinking budget with nothing left for the actual call) — this used to
    crash with an unhandled TypeError indexing into None. Must instead
    return {} so call_structured's existing retry loop treats it as an
    ordinary validation failure."""

    class _FakeMessage:
        tool_calls = None

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeUsage:
        total_tokens = 42

    class _FakeResponse:
        choices = [_FakeChoice()]
        usage = _FakeUsage()

    async def fake_acompletion(**kwargs):
        return _FakeResponse()

    import committee.llm.providers.litellm_provider as provider_module

    monkeypatch.setattr(provider_module.litellm, "acompletion", fake_acompletion)

    caller = LiteLLMRawCaller(model="vertex_ai/gemini-2.5-flash", api_key="fake-key", timeout_seconds=60)
    raw_args, tokens_used = await caller(
        system_prompt="sys", user_prompt="user", schema={}, retry_note=None, max_tokens=500
    )

    assert raw_args == {}
    assert tokens_used == 42


async def test_openai_raw_caller_returns_empty_dict_when_no_tool_call_in_response():
    from committee.llm.providers.openai_provider import OpenAIRawCaller

    caller = OpenAIRawCaller.__new__(OpenAIRawCaller)
    caller.model = "gpt-4o"
    caller.temperature = None

    class _FakeMessage:
        tool_calls = None

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeUsage:
        total_tokens = 30

    class _FakeResponse:
        choices = [_FakeChoice()]
        usage = _FakeUsage()

    class _FakeCompletions:
        async def create(self, **kwargs):
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    caller._client = _FakeClient()

    raw_args, tokens_used = await caller(
        system_prompt="sys", user_prompt="user", schema={}, retry_note=None, max_tokens=500
    )

    assert raw_args == {}
    assert tokens_used == 30


async def test_anthropic_raw_caller_returns_empty_dict_when_no_tool_use_block():
    from committee.llm.providers.anthropic_provider import AnthropicRawCaller

    caller = AnthropicRawCaller.__new__(AnthropicRawCaller)
    caller.model = "claude-sonnet-4-6"
    caller.temperature = None

    class _FakeTextBlock:
        type = "text"

    class _FakeUsage:
        input_tokens = 10
        output_tokens = 20

    class _FakeResponse:
        content = [_FakeTextBlock()]  # no tool_use block at all
        usage = _FakeUsage()

    class _FakeMessages:
        async def create(self, **kwargs):
            return _FakeResponse()

    class _FakeClient:
        messages = _FakeMessages()

    caller._client = _FakeClient()

    raw_args, tokens_used = await caller(
        system_prompt="sys", user_prompt="user", schema={}, retry_note=None, max_tokens=500
    )

    assert raw_args == {}
    assert tokens_used == 30


async def test_litellm_raw_caller_ollama_falls_back_to_default_base_url(monkeypatch):
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop before a real network call")

    import committee.llm.providers.litellm_provider as provider_module

    monkeypatch.setattr(provider_module.litellm, "acompletion", fake_acompletion)

    caller = LiteLLMRawCaller(model="ollama/llama3.1", api_key="", timeout_seconds=60)
    try:
        await caller(system_prompt="sys", user_prompt="user", schema={}, retry_note=None, max_tokens=500)
    except RuntimeError:
        pass

    assert captured["api_base"] == "http://localhost:11434"


async def test_anthropic_raw_caller_passes_temperature():
    from committee.llm.providers.anthropic_provider import AnthropicRawCaller

    caller = AnthropicRawCaller.__new__(AnthropicRawCaller)
    caller.model = "claude-sonnet-4-6"
    caller.temperature = 0.5

    captured = {}

    class _FakeContentBlock:
        type = "tool_use"
        input = {"value": "ok"}

    class _FakeUsage:
        input_tokens = 10
        output_tokens = 5

    class _FakeResponse:
        content = [_FakeContentBlock()]
        usage = _FakeUsage()

    class _FakeMessages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResponse()

    class _FakeClient:
        messages = _FakeMessages()

    caller._client = _FakeClient()

    await caller(system_prompt="sys", user_prompt="user", schema={}, retry_note=None, max_tokens=500)

    assert captured["temperature"] == 0.5


async def test_openai_raw_caller_passes_temperature():
    from committee.llm.providers.openai_provider import OpenAIRawCaller

    caller = OpenAIRawCaller.__new__(OpenAIRawCaller)
    caller.model = "gpt-4o"
    caller.temperature = 0.7

    captured = {}

    class _FakeFunction:
        arguments = '{"value": "ok"}'

    class _FakeToolCall:
        function = _FakeFunction()

    class _FakeMessage:
        tool_calls = [_FakeToolCall()]

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeUsage:
        total_tokens = 15

    class _FakeResponse:
        choices = [_FakeChoice()]
        usage = _FakeUsage()

    class _FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    caller._client = _FakeClient()

    await caller(system_prompt="sys", user_prompt="user", schema={}, retry_note=None, max_tokens=500)

    assert captured["temperature"] == 0.7
