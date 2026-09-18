from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM ---
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    llm_api_key: str = ""
    llm_max_retries: int = 3
    llm_timeout_seconds: int = 60
    # Base delay for exponential backoff between retries of the *same*
    # (primary) provider on a transient error (429/503) — attempt N sleeps
    # roughly llm_retry_backoff_seconds * 2**(N-1) before trying again.
    # Separate from llm_max_retries (which also governs validation retries,
    # see structured_output.py) since a transient-error retry needs a delay
    # and validation retries don't.
    llm_retry_backoff_seconds: float = 1.0

    # --- LLM fallback (optional) ---
    # All unset (None) by default: no fallback configured, primary-only
    # behavior is unchanged. When set, a fallback provider is only ever
    # tried after llm_fallback_max_retries attempts against the primary have
    # exhausted on a transient error (429/503) — never on a non-transient
    # one (auth, bad request), and never as a first choice.
    llm_fallback_provider: str | None = None
    llm_fallback_model: str | None = None
    llm_fallback_api_key: str | None = None
    llm_fallback_max_retries: int = 2

    # Vertex AI service-account auth only (LLM_PROVIDER=litellm, LLM_MODEL=vertex_ai/...).
    # Unused by anthropic/openai and by litellm's plain-API-key routes (e.g. gemini/...).
    # GOOGLE_APPLICATION_CREDENTIALS itself is read directly from the process
    # environment by Google's ADC machinery, not through this Settings object.
    llm_vertex_project: str | None = None
    llm_vertex_location: str | None = None

    # Ollama (local models) via litellm — only used when LLM_MODEL (or a
    # per-debate override) starts with "ollama/". Ollama's own default port;
    # override if it's running elsewhere (a different host, a container).
    ollama_base_url: str = "http://localhost:11434"

    # --- Debate defaults ---
    default_token_budget: int = 50_000
    default_num_rounds: int = 3
    convergence_high_threshold: float = 0.75
    convergence_low_threshold: float = 0.4
    conflict_resolution_strategy: str = "flag_unresolved"
    # Minimum tokens BudgetManager.allocate() guarantees each agent per
    # round, even after exploit-mode reallocation shrinks non-contested
    # agents' share — see budget_manager.py's MIN_VIABLE_ALLOCATION. Default
    # is a modest, provider-agnostic floor (matches
    # structured_output.py's own MIN_MAX_TOKENS) that's enough for any
    # provider's call_structured floor clamp to have room, without being so
    # high it blocks a normal debate against a capable model. Raise this
    # per-debate (DebateConfig.min_viable_allocation_per_agent) when running
    # against a weaker/local model that needs more headroom in practice —
    # this is not something one hardcoded constant can get right for every
    # provider.
    min_viable_allocation_per_agent: int = 256

    # --- Storage ---
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "investment_committee"
    redis_url: str = "redis://localhost:6379/0"
    trace_json_dir: str = "./traces"
    # How long a pod's claim on a run_id survives without a heartbeat before
    # another pod may reclaim it (crash recovery for the cross-pod run lock
    # — see storage/run_lock_store.py). Refreshed once per debate round, so
    # this only needs to exceed real round latency with margin, not be tight.
    run_lock_lease_seconds: int = 120

    # --- Observability ---
    mlflow_tracking_uri: str = "./mlruns"
    prometheus_port: int = 9100
    log_level: str = "INFO"
    log_format: str = "json"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000


def get_settings() -> Settings:
    return Settings()
