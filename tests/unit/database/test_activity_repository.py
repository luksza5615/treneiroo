from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from garmin_buddy.database.db_service import ActivityRepository


class _DummyConnection:
    def __enter__(self) -> "_DummyConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _DummyDatabase:
    def get_db_connection(self) -> _DummyConnection:
        return _DummyConnection()


def test_get_training_summary_returns_expected_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_read_sql_query(sql, conn, params=None):
        captured["query"] = str(sql)
        captured["params"] = params
        return pd.DataFrame(
            [
                {
                    "activities_count": 3,
                    "distance_km": 42.5,
                    "avg_hr": 145.2,
                    "calories_burnt": 2100,
                    "ascent_m": 512,
                    "aerobic_training_effect_0_to_5": 3.2,
                    "anaerobic_training_effect_0_to_5": 1.1,
                }
            ]
        )

    monkeypatch.setattr(pd, "read_sql_query", _fake_read_sql_query)
    repo = ActivityRepository(_DummyDatabase())

    summary = repo.get_training_summary(date(2026, 1, 1), date(2026, 1, 7))

    assert summary["activities_count"] == pytest.approx(3.0)
    assert summary["distance_km"] == pytest.approx(42.5)
    assert "activity_date >= :start_date" in captured["query"]
    assert captured["params"] == {
        "start_date": date(2026, 1, 1),
        "end_date": date(2026, 1, 7),
    }


def test_list_key_sessions_orders_by_total_training_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_read_sql_query(sql, conn, params=None):
        captured["query"] = str(sql)
        captured["params"] = params
        return pd.DataFrame()

    monkeypatch.setattr(pd, "read_sql_query", _fake_read_sql_query)
    repo = ActivityRepository(_DummyDatabase())

    repo.list_key_sessions(date(2026, 1, 1), date(2026, 1, 7), n=3)

    assert "TOP 3" in captured["query"]
    assert "COALESCE(aerobic_training_effect_0_to_5, 0)" in captured["query"]
    assert "COALESCE(anaerobic_training_effect_0_to_5, 0)" in captured["query"]
    assert captured["params"] == {
        "start_date": date(2026, 1, 1),
        "end_date": date(2026, 1, 7),
    }


def test_get_training_summary_rejects_invalid_range() -> None:
    repo = ActivityRepository(_DummyDatabase())

    with pytest.raises(ValueError, match="Start date cannot be later than end date"):
        repo.get_training_summary(date(2026, 1, 7), date(2026, 1, 1))


def test_list_key_sessions_rejects_invalid_limit() -> None:
    repo = ActivityRepository(_DummyDatabase())

    with pytest.raises(ValueError, match="n must be a positive integer"):
        repo.list_key_sessions(date(2026, 1, 1), date(2026, 1, 7), n=0)
