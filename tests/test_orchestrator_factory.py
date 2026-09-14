from committee.config import Settings
from committee.models.requests import DebateConfig
from committee.orchestrator_factory import build_orchestrator


def _settings(**overrides) -> Settings:
    return Settings(llm_provider="anthropic", llm_model="claude-sonnet-4-6", llm_api_key="fake", **overrides)


def test_build_orchestrator_falls_back_to_settings_thresholds_when_config_leaves_them_unset():
    """Real gap found while wiring convergence thresholds up as a frontend
    override: convergence_low_threshold/convergence_high_threshold used to
    be plain (non-Optional) floats on DebateConfig with their own hardcoded
    defaults (0.4/0.75) — Settings.convergence_low_threshold/
    convergence_high_threshold existed but were silently dead, never read
    by build_orchestrator, and there was no way to represent "use the
    server default" over the wire (sending null 422'd). Now None means
    exactly that, matching every other llm_* override's pattern."""
    settings = _settings(convergence_low_threshold=0.3, convergence_high_threshold=0.6)
    config = DebateConfig(total_token_budget=8000, num_rounds=2)

    orchestrator = build_orchestrator(settings, config, enable_mongo=False, enable_redis=False)

    assert orchestrator.controller.low_threshold == 0.3
    assert orchestrator.controller.high_threshold == 0.6


def test_build_orchestrator_uses_config_thresholds_when_explicitly_set():
    settings = _settings(convergence_low_threshold=0.3, convergence_high_threshold=0.6)
    config = DebateConfig(
        total_token_budget=8000,
        num_rounds=2,
        convergence_low_threshold=0.1,
        convergence_high_threshold=0.45,
    )

    orchestrator = build_orchestrator(settings, config, enable_mongo=False, enable_redis=False)

    assert orchestrator.controller.low_threshold == 0.1
    assert orchestrator.controller.high_threshold == 0.45
