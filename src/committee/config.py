from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM ---
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    llm_api_key: str = ""
    llm_max_retries: int = 3
    llm_timeout_seconds: int = 60

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

    # --- Storage ---
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "investment_committee"
    redis_url: str = "redis://localhost:6379/0"
    trace_json_dir: str = "./traces"

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
