from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from garmin_buddy.ai.tools.training_plan_preparation_tools import (
    PreparationToolRegistry,
    ToolValidationError,
)


class _DummyRepository:
    def get_execution_summary(self, start_date, end_date, *, athlete_id=None):
        return {"executed_sessions": 3, "distance_km": 32.0}

    def list_key_sessions(self, start_date, end_date, *, athlete_id=None, n=5):
        return pd.DataFrame([{"activity_id": 123, "activity_date": datetime(2026, 2, 1)}])

    def compare_planned_vs_executed(
        self, planned_sessions, start_date, end_date, *, athlete_id=None
    ):
        return {"planned_sessions": len(planned_sessions), "adherence_rate": 1.0}


def test_preparation_tool_registry_caches_duplicate_requests() -> None:
    registry = PreparationToolRegistry(
        repository=_DummyRepository(),
        max_tool_calls=2,
        profile_loader=lambda: {"goals": ["Goal"], "availability": [], "constraints": [], "preferences": [], "injury_notes": [], "source_notes": []},
    )

    first = registry.call_tool("get_runner_profile", {})
    second = registry.call_tool("get_runner_profile", {})

    assert first.ok is True
    assert second.ok is True
    assert registry.get_call_log()[1]["cached"] is True


def test_preparation_tool_registry_compares_planned_vs_executed() -> None:
    registry = PreparationToolRegistry(
        repository=_DummyRepository(),
        max_tool_calls=4,
        training_log_loader=lambda start, end: [{"date": "2026-02-01", "session_type": "easy", "planned_focus": "aerobic", "planned_load": "low", "notes": ""}],
    )

    result = registry.call_tool(
        "compare_planned_vs_executed",
        {"start_date": date(2026, 2, 1), "end_date": date(2026, 2, 7)},
    )

    assert result.ok is True
    assert result.payload["planned_sessions"] == 1


def test_preparation_tool_registry_rejects_unknown_tool() -> None:
    registry = PreparationToolRegistry(repository=_DummyRepository(), max_tool_calls=1)

    with pytest.raises(ToolValidationError, match="allowlist"):
        registry.call_tool("unknown", {})
