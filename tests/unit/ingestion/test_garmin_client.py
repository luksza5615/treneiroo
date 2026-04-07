from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import Mock

import pytest
from garminconnect import GarminConnectTooManyRequestsError

from garmin_buddy.ingestion import garmin_client as garmin_client_module
from garmin_buddy.ingestion.garmin_client import GarminClient, GarminRateLimitError


class _FakeGarmin:
    def __init__(self, email: str, password: str) -> None:
        self.email = email
        self.password = password
        self.garth = Mock()
        self.login = Mock()
        self.get_activities_by_date = Mock()


def test_login_to_garmin_uses_tokenstore_and_persists_tokens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_client = _FakeGarmin("user@example.com", "secret")
    garmin_ctor = Mock(return_value=fake_client)
    monkeypatch.setattr(garmin_client_module, "Garmin", garmin_ctor)

    client = GarminClient(
        "user@example.com",
        "secret",
        tokenstore_path=tmp_path / ".garmin_session",
    )

    client.login_to_garmin()

    garmin_ctor.assert_called_once_with("user@example.com", "secret")
    fake_client.garth.configure.assert_called_once_with(
        status_forcelist=(408, 500, 502, 503, 504)
    )
    fake_client.login.assert_called_once_with(tokenstore=str(tmp_path / ".garmin_session"))
    fake_client.garth.dump.assert_called_once_with(str(tmp_path / ".garmin_session"))


def test_login_to_garmin_falls_back_to_credentials_when_tokenstore_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_client = _FakeGarmin("user@example.com", "secret")
    fake_client.login.side_effect = [FileNotFoundError("missing"), None]
    monkeypatch.setattr(garmin_client_module, "Garmin", Mock(return_value=fake_client))

    client = GarminClient(
        "user@example.com",
        "secret",
        tokenstore_path=tmp_path / ".garmin_session",
    )

    client.login_to_garmin()

    assert fake_client.login.call_args_list[0].kwargs == {
        "tokenstore": str(tmp_path / ".garmin_session")
    }
    assert fake_client.login.call_args_list[1].args == ()
    assert fake_client.login.call_args_list[1].kwargs == {}
    fake_client.garth.dump.assert_called_once_with(str(tmp_path / ".garmin_session"))


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
