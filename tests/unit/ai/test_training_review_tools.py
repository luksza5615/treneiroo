from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from garmin_buddy.ai.tools.training_review_tools import ToolRegistry, ToolValidationError


class _DummyRepository:
    def __init__(self) -> None:
        self.summary_calls: list[tuple] = []
        self.key_session_calls: list[tuple] = []
        self.activity_calls: list[int] = []

    def get_training_summary(self, start_date, end_date, *, athlete_id=None):
        self.summary_calls.append((start_date, end_date, athlete_id))
        return {"activities_count": 1.0}

    def list_key_sessions(self, start_date, end_date, *, athlete_id=None, n=5):
        self.key_session_calls.append((start_date, end_date, athlete_id, n))
        return pd.DataFrame([{"activity_id": 123}])

    def get_activity_by_id(self, activity_id: int):
        self.activity_calls.append(activity_id)
        return pd.DataFrame([{"activity_id": activity_id}])


def test_call_tool_rejects_unknown_tool() -> None:
    registry = ToolRegistry(_DummyRepository(), max_tool_calls=1)

    with pytest.raises(ToolValidationError, match="allowlist"):
        registry.call_tool("unknown", {})


def test_call_tool_caches_duplicate_requests() -> None:
    repo = _DummyRepository()
    registry = ToolRegistry(repo, max_tool_calls=2)

    result_1 = registry.call_tool(
        "get_training_summary",
        {"start_date": "2026-01-01", "end_date": "2026-01-07"},
    )
    result_2 = registry.call_tool(
        "get_training_summary",
        {"start_date": "2026-01-01", "end_date": "2026-01-07"},
    )

    assert result_1.ok is True
    assert result_2.ok is True
    assert len(repo.summary_calls) == 1


def test_call_tool_enforces_budget() -> None:
    registry = ToolRegistry(_DummyRepository(), max_tool_calls=1)

    registry.call_tool(
        "get_training_summary",
        {"start_date": "2026-01-01", "end_date": "2026-01-07"},
    )
    result = registry.call_tool(
        "list_key_sessions",
        {"start_date": "2026-01-01", "end_date": "2026-01-07"},
    )

    assert result.ok is False
    assert result.error == "Tool call budget exceeded."


def test_list_key_sessions_validates_n() -> None:
    registry = ToolRegistry(_DummyRepository(), max_tool_calls=1)

    with pytest.raises(ToolValidationError, match="n must be a positive integer"):
        registry.call_tool(
            "list_key_sessions",
            {"start_date": "2026-01-01", "end_date": "2026-01-07", "n": 0},
        )


def test_get_activity_requires_positive_id() -> None:
    registry = ToolRegistry(_DummyRepository(), max_tool_calls=1)

    with pytest.raises(ToolValidationError, match="activity_id must be a positive"):
        registry.call_tool("get_activity", {"activity_id": -1})


def test_call_tool_accepts_date_objects() -> None:
    repo = _DummyRepository()
    registry = ToolRegistry(repo, max_tool_calls=1)

    registry.call_tool(
        "get_training_summary",
        {"start_date": date(2026, 1, 1), "end_date": date(2026, 1, 7)},
    )

    assert repo.summary_calls


def test_call_tool_logs_cache_hits() -> None:
    registry = ToolRegistry(_DummyRepository(), max_tool_calls=2)

    registry.call_tool(
        "get_training_summary",
        {"start_date": "2026-01-01", "end_date": "2026-01-07"},
    )
    registry.call_tool(
        "get_training_summary",
        {"start_date": "2026-01-01", "end_date": "2026-01-07"},
    )

    log = registry.get_call_log()
    assert len(log) == 2
    assert log[1]["cached"] is True
