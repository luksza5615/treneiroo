from __future__ import annotations

from datetime import date

import pandas as pd

from garmin_buddy.ai.contracts import TrainingReviewReport
from garmin_buddy.ai.tools.training_review_tools import ToolRegistry
from garmin_buddy.ai.workflows.training_review import (
    TrainingReviewInputs,
    run_training_review,
)


class _DummyRepository:
    def get_training_summary(self, start_date, end_date, *, athlete_id=None):
        return {"activities_count": 2.0}

    def list_key_sessions(self, start_date, end_date, *, athlete_id=None, n=5):
        return pd.DataFrame(
            [
                {
                    "activity_id": 123,
                    "activity_date": date(2026, 1, 2),
                    "sport": "running",
                }
            ]
        )

    def get_activity_by_id(self, activity_id: int):
        return pd.DataFrame()


class _FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._responses.pop(0)


def _valid_report_json() -> str:
    return (
        "{"
        '"headline":"Solid week.",'
        '"positives":["Consistent volume."],'
        '"risks":["Fatigue risk."],'
        '"priorities_next_7_days":["Rest day.","Easy run.","Mobility work."],'
        '"evidence":["2026-01-02 activity:123 Long run"],'
        '"confidence":0.6,'
        '"missing_data":[],'
        '"disclaimer":"This report is informational only and is not medical advice."'
        "}"
    )


def test_run_training_review_happy_path() -> None:
    llm = _FakeLLM([_valid_report_json()])
    registry = ToolRegistry(_DummyRepository(), max_tool_calls=2)
    inputs = TrainingReviewInputs(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 7),
    )

    result = run_training_review(
        llm_client=llm,
        tool_registry=registry,
        inputs=inputs,
    )

    assert result.parse_ok is True
    assert isinstance(result.report, TrainingReviewReport)
    assert result.report.confidence == 0.6


def test_run_training_review_repairs_invalid_json() -> None:
    llm = _FakeLLM(["{invalid json", _valid_report_json()])
    registry = ToolRegistry(_DummyRepository(), max_tool_calls=2)
    inputs = TrainingReviewInputs(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 7),
    )

    result = run_training_review(
        llm_client=llm,
        tool_registry=registry,
        inputs=inputs,
    )

    assert result.parse_ok is True
    assert len(llm.calls) == 2


def test_run_training_review_returns_fallback_when_repair_fails() -> None:
    llm = _FakeLLM(["{invalid json", "{still invalid"])
    registry = ToolRegistry(_DummyRepository(), max_tool_calls=2)
    inputs = TrainingReviewInputs(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 7),
    )

    result = run_training_review(
        llm_client=llm,
        tool_registry=registry,
        inputs=inputs,
    )

    assert result.parse_ok is False
    assert result.report.confidence == 0.0
