from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import pandas as pd
import pytest

from garmin_buddy.ai.contracts.contracts import TrainingReviewReport
from garmin_buddy.ai.logging.run_store import RunStore
from garmin_buddy.ai.tools.training_review_tools import ToolRegistry
from garmin_buddy.ai.workflows.training_review import (
    TrainingReviewInputs,
    _maybe_fetch_evidence,
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
        self.calls: list[dict[str, str | None]] = []

    def generate(
        self, prompt: str, *, system_instruction: str | None = None
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system_instruction": system_instruction,
            }
        )
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
        '"missing_data":[]'
        "}"
    )


def test_run_training_review_happy_path() -> None:
    llm = _FakeLLM(['{"activity_ids":[123]}', _valid_report_json()])
    registry = ToolRegistry(_DummyRepository(), max_tool_calls=3)
    store = RunStore(_workspace_temp_dir("review-success"))
    inputs = TrainingReviewInputs(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 7),
    )

    result = run_training_review(
        llm_client=llm,
        tool_registry=registry,
        inputs=inputs,
        run_store=store,
        model_name="test-model",
    )

    assert result.parse_ok is True
    assert isinstance(result.report, TrainingReviewReport)
    assert result.report.confidence == 0.6
    assert result.run_id is not None
    assert llm.calls[1]["system_instruction"] is not None
    assert "Produce only one JSON object." in llm.calls[1]["system_instruction"]
    assert "Generate a TrainingReviewReport for:" in llm.calls[1]["prompt"]
    assert "Training summary:" in llm.calls[1]["prompt"]
    assert '"evidence_sessions": []' not in llm.calls[1]["prompt"]


def test_run_training_review_repairs_invalid_json() -> None:
    llm = _FakeLLM(['{"activity_ids":[123]}', "{invalid json", _valid_report_json()])
    registry = ToolRegistry(_DummyRepository(), max_tool_calls=3)
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
    assert len(llm.calls) == 3


def test_run_training_review_accepts_fenced_json_response() -> None:
    llm = _FakeLLM(['{"activity_ids":[123]}', f"```json\n{_valid_report_json()}\n```"])
    registry = ToolRegistry(_DummyRepository(), max_tool_calls=3)
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
    assert result.report.confidence == 0.6
    assert len(llm.calls) == 2


def test_run_training_review_returns_fallback_when_repair_fails() -> None:
    llm = _FakeLLM(['{"activity_ids":[123]}', "{invalid json", "{still invalid"])
    registry = ToolRegistry(_DummyRepository(), max_tool_calls=3)
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


def test_maybe_fetch_evidence_accepts_fenced_json_response() -> None:
    llm = _FakeLLM(['```json\n{"activity_ids":[123]}\n```'])
    registry = ToolRegistry(_DummyRepository(), max_tool_calls=3)
    missing_data: list[str] = []

    evidence = _maybe_fetch_evidence(
        llm_client=llm,
        tool_registry=registry,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 7),
        athlete_id=None,
        key_sessions=[
            {
                "activity_id": 123,
                "activity_date": date(2026, 1, 2),
                "sport": "running",
            }
        ],
        missing_data=missing_data,
    )

    assert evidence == []
    assert missing_data == []


def test_run_training_review_persists_failed_runs_when_llm_errors() -> None:
    llm = _RaisingLLM("503 Service Unavailable")
    registry = ToolRegistry(_DummyRepository(), max_tool_calls=3)
    temp_dir = _workspace_temp_dir("review-failure")
    store = RunStore(temp_dir)
    inputs = TrainingReviewInputs(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 7),
        include_key_sessions=False,
    )

    with pytest.raises(RuntimeError, match="503 Service Unavailable"):
        run_training_review(
            llm_client=llm,
            tool_registry=registry,
            inputs=inputs,
            run_store=store,
            model_name="test-model",
        )

    saved_line = json.loads(
        (temp_dir / "training_review_runs.jsonl").read_text(encoding="utf-8").strip()
    )
    assert saved_line["run_status"] == "failed"
    assert saved_line["failed_stage"] == "generate_report"
    assert saved_line["error"]["type"] == "RuntimeError"


class _RaisingLLM:
    def __init__(self, message: str) -> None:
        self.message = message

    def generate(
        self, prompt: str, *, system_instruction: str | None = None
    ) -> str:
        raise RuntimeError(self.message)


def _workspace_temp_dir(name: str) -> Path:
    from uuid import uuid4

    path = Path("zignored") / "pytest-temp" / f"{name}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
