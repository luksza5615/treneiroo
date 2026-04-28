from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pandas as pd
import pytest

from garmin_buddy.ai.contracts.contracts import TrainingReviewReport
from garmin_buddy.ai.llm_analysis_service import TokenUsageTotals
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
    def __init__(
        self,
        responses: list[str | Exception],
        usages: list[tuple[int, int, int] | None] | None = None,
    ) -> None:
        self._responses = responses
        self._usages = list(usages or [])
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        response_json_schema: Mapping[str, Any] | None = None,
        usage_tracker: TokenUsageTotals | None = None,
    ) -> str:
        usage = self._usages.pop(0) if self._usages else None
        if usage_tracker is not None and usage is not None:
            usage_tracker.add_usage(_usage_metadata(*usage))
        self.calls.append(
            {
                "prompt": prompt,
                "system_instruction": system_instruction,
                "response_json_schema": response_json_schema,
            }
        )
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _valid_report_json() -> str:
    return (
        "{"
        '"summary":"Solid week.",'
        '"positives":["Consistent volume."],'
        '"mistakes":["Fatigue risk."],'
        '"main_lessons_and_recommendations":["Rest day.","Easy run."],'
        '"evidence":["2026-01-02 activity:123 Long run"],'
        '"confidence":0.6,'
        '"missing_data":[]'
        "}"
    )


def test_run_training_review_happy_path() -> None:
    llm = _FakeLLM(
        ['{"activity_ids":[123]}', _valid_report_json()],
        usages=[(11, 0, 2), (23, 0, 17)],
    )
    registry = ToolRegistry(_DummyRepository(), max_tool_calls=3)
    temp_dir = _workspace_temp_dir("review-success")
    store = RunStore(temp_dir)
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
    assert llm.calls[0]["response_json_schema"]["required"] == ["activity_ids"]
    assert llm.calls[1]["system_instruction"] is not None
    assert "summary" in llm.calls[1]["response_json_schema"]["required"]
    assert "Produce only one JSON object." in llm.calls[1]["system_instruction"]
    assert "Produce the most insightful training review" in llm.calls[1]["prompt"]
    assert "Training summary:" in llm.calls[1]["prompt"]
    assert '"evidence_sessions": []' not in llm.calls[1]["prompt"]
    saved_line = json.loads(
        (temp_dir / "training_review_runs.jsonl").read_text(encoding="utf-8").strip()
    )
    assert saved_line["total_input_tokens"] == 34
    assert saved_line["total_output_tokens"] == 19


def test_run_training_review_uses_fixed_budget_without_budget_missing_data() -> None:
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
    assert len(llm.calls) == 1
    assert "Key sessions:" in llm.calls[0]["prompt"]
    assert "evidence_tool_budget_exhausted" not in llm.calls[0]["prompt"]


def test_run_training_review_repairs_invalid_json() -> None:
    llm = _FakeLLM(
        ['{"activity_ids":[123]}', "{invalid json", _valid_report_json()],
        usages=[(7, 0, 2), (19, 0, 8), (13, 0, 11)],
    )
    registry = ToolRegistry(_DummyRepository(), max_tool_calls=3)
    temp_dir = _workspace_temp_dir("review-repair")
    inputs = TrainingReviewInputs(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 7),
    )

    result = run_training_review(
        llm_client=llm,
        tool_registry=registry,
        inputs=inputs,
        run_store=RunStore(temp_dir),
    )

    assert result.parse_ok is True
    assert len(llm.calls) == 3
    assert llm.calls[2]["system_instruction"] == "Return valid JSON only."
    assert "summary" in llm.calls[2]["response_json_schema"]["required"]
    saved_line = json.loads(
        (temp_dir / "training_review_runs.jsonl").read_text(encoding="utf-8").strip()
    )
    assert saved_line["total_input_tokens"] == 39
    assert saved_line["total_output_tokens"] == 21


def test_run_training_review_repairs_fenced_json_response() -> None:
    llm = _FakeLLM(
        [
            '{"activity_ids":[123]}',
            f"```json\n{_valid_report_json()}\n```",
            _valid_report_json(),
        ]
    )
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
    assert len(llm.calls) == 3
    assert "Fix the following JSON" in llm.calls[2]["prompt"]


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


def test_maybe_fetch_evidence_rejects_fenced_json_response() -> None:
    llm = _FakeLLM(['```json\n{"activity_ids":[123]}\n```'])
    registry = ToolRegistry(_DummyRepository(), max_tool_calls=3)
    missing_data: list[str] = []
    usage_tracker = TokenUsageTotals()

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
        usage_tracker=usage_tracker,
    )

    assert evidence == []
    assert missing_data == ["evidence_request_invalid_json"]
    assert llm.calls[0]["response_json_schema"]["required"] == ["activity_ids"]


def test_run_training_review_persists_failed_runs_when_llm_errors() -> None:
    llm = _FakeLLM(
        ['{"activity_ids":[123]}', RuntimeError("503 Service Unavailable")],
        usages=[(9, 0, 2)],
    )
    registry = ToolRegistry(_DummyRepository(), max_tool_calls=3)
    temp_dir = _workspace_temp_dir("review-failure")
    store = RunStore(temp_dir)
    inputs = TrainingReviewInputs(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 7),
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
    assert saved_line["total_input_tokens"] == 9
    assert saved_line["total_output_tokens"] == 2


def _workspace_temp_dir(name: str) -> Path:
    from uuid import uuid4

    path = Path("zignored") / "pytest-temp" / f"{name}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _usage_metadata(
    prompt_tokens: int,
    tool_use_prompt_tokens: int,
    candidate_tokens: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_token_count=prompt_tokens,
        tool_use_prompt_token_count=tool_use_prompt_tokens,
        candidates_token_count=candidate_tokens,
    )
