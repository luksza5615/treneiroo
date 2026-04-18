from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import Mock, call

import pytest
from garminconnect import GarminConnectTooManyRequestsError

from garmin_buddy.ingestion import garmin_client as garmin_client_module
from garmin_buddy.ingestion.garmin_client import (
    GarminClient,
    GarminMFARequiredError,
    GarminRateLimitError,
)


class _FakeGarmin:
    def __init__(self) -> None:
        self.login = Mock()
        self.get_activities_by_date = Mock()


def test_login_to_garmin_restores_saved_tokens_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_client = _FakeGarmin()
    garmin_ctor = Mock(return_value=fake_client)
    monkeypatch.setattr(garmin_client_module, "Garmin", garmin_ctor)

    client = GarminClient(
        "user@example.com",
        "secret",
        tokenstore_path=tmp_path / ".garmin_session",
    )

    client.login_to_garmin()

    garmin_ctor.assert_called_once_with()
    fake_client.login.assert_called_once_with(str(tmp_path / ".garmin_session"))


def test_login_to_garmin_falls_back_to_credentials_when_saved_tokens_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    restored_client = _FakeGarmin()
    restored_client.login.side_effect = FileNotFoundError("missing")
    fresh_client = _FakeGarmin()
    prompt_mfa = Mock(return_value="123456")
    garmin_ctor = Mock(side_effect=[restored_client, fresh_client])
    monkeypatch.setattr(garmin_client_module, "Garmin", garmin_ctor)

    client = GarminClient(
        "user@example.com",
        "secret",
        tokenstore_path=tmp_path / ".garmin_session",
        prompt_mfa=prompt_mfa,
    )

    client.login_to_garmin()

    assert garmin_ctor.call_args_list == [
        call(),
        call(
            email="user@example.com",
            password="secret",
            prompt_mfa=prompt_mfa,
        ),
    ]
    restored_client.login.assert_called_once_with(str(tmp_path / ".garmin_session"))
    fresh_client.login.assert_called_once_with(str(tmp_path / ".garmin_session"))


def test_default_prompt_mfa_requires_interactive_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeStdin:
        @staticmethod
        def isatty() -> bool:
            return False

    monkeypatch.setattr(garmin_client_module.sys, "stdin", _FakeStdin())

    client = GarminClient("user@example.com", "secret")

    with pytest.raises(GarminMFARequiredError, match="Configure an MFA callback"):
        client._default_prompt_mfa()


def test_get_garmin_activities_history_raises_rate_limit_error() -> None:
    client = GarminClient("user@example.com", "secret")
    logged_in_client = Mock()
    logged_in_client.get_activities_by_date.side_effect = GarminConnectTooManyRequestsError(
        "rate limited"
    )
    client._client = logged_in_client

    with pytest.raises(GarminRateLimitError, match="rate-limited activity history"):
        client.get_garmin_activities_history(
            start_date=date(2026, 3, 1), end_date=date(2026, 3, 2)
        )
