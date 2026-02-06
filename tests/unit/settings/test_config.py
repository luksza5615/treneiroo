import pytest

from garmin_buddy.settings.config import Config


def test_config_feature_flag_parses_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIT_DIR_PATH", "C:\\tmp")
    monkeypatch.setenv("GARMIN_EMAIL", "user@example.com")
    monkeypatch.setenv("GARMIN_PASSWORD", "pass")
    monkeypatch.setenv("DB_CONNECTION_STRING", "Server=.;Database=test;")
    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("FEATURE_TRAINING_REVIEW", "true")

    cfg = Config.from_env()

    assert cfg.feature_training_review is True


def test_config_feature_flag_invalid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIT_DIR_PATH", "C:\\tmp")
    monkeypatch.setenv("GARMIN_EMAIL", "user@example.com")
    monkeypatch.setenv("GARMIN_PASSWORD", "pass")
    monkeypatch.setenv("DB_CONNECTION_STRING", "Server=.;Database=test;")
    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("FEATURE_TRAINING_REVIEW", "maybe")

    with pytest.raises(ValueError, match="Invalid boolean value"):
        Config.from_env()
