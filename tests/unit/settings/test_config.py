from pathlib import Path

import pytest

from garmin_buddy.settings.config import Config


def test_config_direct_init_uses_optional_defaults() -> None:
    cfg = Config(
        fit_dir_path=Path("C:\\tmp"),
        garmin_email="user@example.com",
        garmin_password="pass",
        db_connection_string="Server=.;Database=test;",
        llm_api_key="key",
    )

    assert cfg.feature_training_review is False
    assert cfg.feature_training_plan_preparation is False
    assert cfg.google_sheets_spreadsheet_id is None
    assert cfg.google_sheets_worksheet_name is None
    assert cfg.google_service_account_info is None


def test_config_feature_flag_parses_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIT_DIR_PATH", "C:\\tmp")
    monkeypatch.setenv("GARMIN_EMAIL", "user@example.com")
    monkeypatch.setenv("GARMIN_PASSWORD", "pass")
    monkeypatch.setenv("DB_CONNECTION_STRING", "Server=.;Database=test;")
    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("FEATURE_TRAINING_REVIEW", "true")
    monkeypatch.setenv("FEATURE_TRAINING_PLAN_PREPARATION", "true")

    cfg = Config.from_env()

    assert cfg.feature_training_review is True
    assert cfg.feature_training_plan_preparation is True


def test_config_feature_flag_invalid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIT_DIR_PATH", "C:\\tmp")
    monkeypatch.setenv("GARMIN_EMAIL", "user@example.com")
    monkeypatch.setenv("GARMIN_PASSWORD", "pass")
    monkeypatch.setenv("DB_CONNECTION_STRING", "Server=.;Database=test;")
    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("FEATURE_TRAINING_REVIEW", "maybe")

    with pytest.raises(ValueError, match="Invalid boolean value"):
        Config.from_env()


def test_config_preparation_flag_invalid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIT_DIR_PATH", "C:\\tmp")
    monkeypatch.setenv("GARMIN_EMAIL", "user@example.com")
    monkeypatch.setenv("GARMIN_PASSWORD", "pass")
    monkeypatch.setenv("DB_CONNECTION_STRING", "Server=.;Database=test;")
    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("FEATURE_TRAINING_PLAN_PREPARATION", "maybe")

    with pytest.raises(ValueError, match="Invalid boolean value"):
        Config.from_env()
