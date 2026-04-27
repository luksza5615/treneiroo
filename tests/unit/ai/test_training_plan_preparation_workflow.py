from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from uuid import uuid4

import pandas as pd
import pytest

from garmin_buddy.ai.logging.preparation_run_store import PreparationRunStore
from garmin_buddy.ai.tools.training_plan_preparation_tools import (
    PreparationToolRegistry,
)
from garmin_buddy.ai.workflows.training_plan_preparation import (
    TrainingPlanPreparationInputs,
    approve_training_plan_strategy,
    generate_phase_plan_from_strategy,
    run_training_plan_preparation,
)


class _DummyRepository:
    def get_execution_summary(self, start_date, end_date, *, athlete_id=None):
        return {"executed_sessions": 4, "distance_km": 42.0, "hard_sessions": 1}

    def list_key_sessions(self, start_date, end_date, *, athlete_id=None, n=5):
        return pd.DataFrame([{"activity_id": 123, "activity_date": date(2026, 2, 1)}])

    def compare_planned_vs_executed(
        self, planned_sessions, start_date, end_date, *, athlete_id=None
    ):
        return {"planned_sessions": len(planned_sessions), "adherence_rate": 0.75}


class _FakeLLM:
    def __init__(
        self,
        responses: list[str | Exception],
        usages: list[tuple[int, int, int] | None] | None = None,
    ) -> None:
        self.responses = responses
        self.usages = list(usages or [])
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        response_json_schema: Mapping[str, Any] | None = None,
        usage_tracker=None,
    ) -> str:
        usage = self.usages.pop(0) if self.usages else None
        if usage_tracker is not None and usage is not None:
            usage_tracker.add_usage(_usage_metadata(*usage))
        self.calls.append(
            {
                "prompt": prompt,
                "system_instruction": system_instruction,
                "response_json_schema": response_json_schema,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _profile_payload(profile_context: str = "Goal: Sub-40 10k") -> dict[str, object]:
    return {
        "profile_context": profile_context,
    }


def _lab_payload(text: str = "Ferritin 28") -> dict[str, object]:
    return {
        "lab_summary": text,
        "lab_markers": {"ferritin": 28.0},
        "lab_fingerprint": text,
    }


def _training_log_loader(start_date, end_date):
    return [
        {
            "date": "2026-02-01",
            "session_type": "easy",
            "planned_focus": "aerobic",
            "planned_load": "low",
            "notes": "",
        }
    ]


def test_run_training_plan_preparation_generates_strategy() -> None:
    llm = _FakeLLM(
        [
            '{"summary":"Lab ok","findings":["Ferritin is low-normal"],"training_implications":["Use conservative intensity"],"risk_flags":["ferritin_watch"],"missing_data":[],"confidence":0.7}',
            '{"summary":"Review","adherence_summary":"Mostly on plan","positive_patterns":["Good consistency"],"execution_issues":["One missed quality day"],"risk_flags":["fatigue_watch"],"missing_data":[],"confidence":0.7}',
            '{"summary":"Synthesis","key_constraints":["Ferritin trend"],"key_opportunities":["Strong consistency"],"planning_priorities":["Stable volume"],"risk_controls":["Watch fatigue"],"assumptions":["No acute injury"],"missing_data":[],"confidence":0.75}',
            '{"objectives":["Durability"],"weekly_frequency":"2x/week","session_focuses":["Single-leg strength"],"integration_notes":["After easy days"],"contraindications":["Reduce on fatigue"],"missing_data":[],"confidence":0.7}',
            '{"planning_horizon":"4 months","strategic_goal":"Build to a strong 10k.","mesocycles":["Base","Specific"],"progression_logic":["Grow load gradually"],"recovery_logic":["Deload every 4th week"],"risks":["Ferritin"],"assumptions":["Good availability"],"missing_data":[],"confidence":0.72}',
        ],
        usages=[(10, 0, 5), (11, 0, 6), (12, 0, 7), (13, 0, 8), (14, 0, 9)],
    )
    temp_dir = _workspace_temp_dir("prep-workflow")
    store = PreparationRunStore(temp_dir)
    registry = PreparationToolRegistry(
        repository=_DummyRepository(),
        max_tool_calls=8,
        training_log_loader=_training_log_loader,
        profile_loader=lambda: _profile_payload(),
        lab_loader=lambda: _lab_payload(),
    )

    result = run_training_plan_preparation(
        llm_client=llm,
        tool_registry=registry,
        run_store=store,
        inputs=TrainingPlanPreparationInputs(
            start_date=date(2026, 2, 1),
            end_date=date(2026, 2, 28),
        ),
    )

    assert result.strategy.approval_status == "pending"
    assert result.phase_plan is None
    assert result.parse_ok is True
    assert llm.calls[0]["response_json_schema"]["required"] == [
        "summary",
        "findings",
        "training_implications",
        "risk_flags",
        "missing_data",
        "confidence",
    ]
    assert llm.calls[4]["response_json_schema"]["required"] == [
        "planning_horizon",
        "strategic_goal",
        "mesocycles",
        "progression_logic",
        "recovery_logic",
        "risks",
        "assumptions",
        "missing_data",
        "confidence",
    ]
    saved_line = json.loads(
        (temp_dir / "training_plan_preparation_runs.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert saved_line["total_input_tokens"] == 60
    assert saved_line["total_output_tokens"] == 35


def test_run_training_plan_preparation_repair_uses_structured_output_schema() -> None:
    llm = _FakeLLM(
        [
            "{invalid json",
            '{"summary":"Lab ok","findings":["Ferritin is low-normal"],"training_implications":["Use conservative intensity"],"risk_flags":["ferritin_watch"],"missing_data":[],"confidence":0.7}',
            '{"summary":"Review","adherence_summary":"Mostly on plan","positive_patterns":["Good consistency"],"execution_issues":["One missed quality day"],"risk_flags":["fatigue_watch"],"missing_data":[],"confidence":0.7}',
            '{"summary":"Synthesis","key_constraints":["Ferritin trend"],"key_opportunities":["Strong consistency"],"planning_priorities":["Stable volume"],"risk_controls":["Watch fatigue"],"assumptions":["No acute injury"],"missing_data":[],"confidence":0.75}',
            '{"objectives":["Durability"],"weekly_frequency":"2x/week","session_focuses":["Single-leg strength"],"integration_notes":["After easy days"],"contraindications":["Reduce on fatigue"],"missing_data":[],"confidence":0.7}',
            '{"planning_horizon":"4 months","strategic_goal":"Build to a strong 10k.","mesocycles":["Base","Specific"],"progression_logic":["Grow load gradually"],"recovery_logic":["Deload every 4th week"],"risks":["Ferritin"],"assumptions":["Good availability"],"missing_data":[],"confidence":0.72}',
        ],
        usages=[(9, 0, 3), (8, 0, 4), (10, 0, 5), (11, 0, 6), (12, 0, 7), (13, 0, 8)],
    )
    temp_dir = _workspace_temp_dir("prep-repair")
    store = PreparationRunStore(temp_dir)
    registry = PreparationToolRegistry(
        repository=_DummyRepository(),
        max_tool_calls=8,
        training_log_loader=_training_log_loader,
        profile_loader=lambda: _profile_payload(),
        lab_loader=lambda: _lab_payload(),
    )

    result = run_training_plan_preparation(
        llm_client=llm,
        tool_registry=registry,
        run_store=store,
        inputs=TrainingPlanPreparationInputs(
            start_date=date(2026, 2, 1),
            end_date=date(2026, 2, 28),
        ),
    )

    assert result.parse_ok is True
    assert result.retry_count == 1
    assert llm.calls[1]["system_instruction"] == "Return valid JSON only."
    assert llm.calls[1]["response_json_schema"] == llm.calls[0][
        "response_json_schema"
    ]
    saved_line = json.loads(
        (temp_dir / "training_plan_preparation_runs.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert saved_line["total_input_tokens"] == 63
    assert saved_line["total_output_tokens"] == 33


def test_generate_phase_plan_marks_strategy_stale_when_inputs_change() -> None:
    llm = _FakeLLM(
        [
            '{"summary":"Lab ok","findings":["Ferritin is low-normal"],"training_implications":["Use conservative intensity"],"risk_flags":["ferritin_watch"],"missing_data":[],"confidence":0.7}',
            '{"summary":"Review","adherence_summary":"Mostly on plan","positive_patterns":["Good consistency"],"execution_issues":["One missed quality day"],"risk_flags":["fatigue_watch"],"missing_data":[],"confidence":0.7}',
            '{"summary":"Synthesis","key_constraints":["Ferritin trend"],"key_opportunities":["Strong consistency"],"planning_priorities":["Stable volume"],"risk_controls":["Watch fatigue"],"assumptions":["No acute injury"],"missing_data":[],"confidence":0.75}',
            '{"objectives":["Durability"],"weekly_frequency":"2x/week","session_focuses":["Single-leg strength"],"integration_notes":["After easy days"],"contraindications":["Reduce on fatigue"],"missing_data":[],"confidence":0.7}',
            '{"planning_horizon":"4 months","strategic_goal":"Build to a strong 10k.","mesocycles":["Base","Specific"],"progression_logic":["Grow load gradually"],"recovery_logic":["Deload every 4th week"],"risks":["Ferritin"],"assumptions":["Good availability"],"missing_data":[],"confidence":0.72}',
        ]
    )
    store = PreparationRunStore(_workspace_temp_dir("prep-workflow"))
    initial_registry = PreparationToolRegistry(
        repository=_DummyRepository(),
        max_tool_calls=8,
        training_log_loader=_training_log_loader,
        profile_loader=lambda: _profile_payload(),
        lab_loader=lambda: _lab_payload(),
    )
    initial_result = run_training_plan_preparation(
        llm_client=llm,
        tool_registry=initial_registry,
        run_store=store,
        inputs=TrainingPlanPreparationInputs(
            start_date=date(2026, 2, 1),
            end_date=date(2026, 2, 28),
        ),
    )
    approve_training_plan_strategy(
        run_store=store, strategy_id=initial_result.strategy.strategy_id
    )

    phase_registry = PreparationToolRegistry(
        repository=_DummyRepository(),
        max_tool_calls=8,
        training_log_loader=_training_log_loader,
        profile_loader=lambda: _profile_payload(profile_context="Goal: Different goal"),
        lab_loader=lambda: _lab_payload(),
    )
    phase_result = generate_phase_plan_from_strategy(
        llm_client=_FakeLLM([]),
        tool_registry=phase_registry,
        run_store=store,
        inputs=TrainingPlanPreparationInputs(
            start_date=date(2026, 2, 1),
            end_date=date(2026, 2, 28),
        ),
        strategy_id=initial_result.strategy.strategy_id,
    )

    assert phase_result.strategy_stale is True
    assert phase_result.strategy.approval_status == "stale"


def test_generate_phase_plan_uses_structured_output_schemas() -> None:
    initial_llm = _FakeLLM(
        [
            '{"summary":"Lab ok","findings":["Ferritin is low-normal"],"training_implications":["Use conservative intensity"],"risk_flags":["ferritin_watch"],"missing_data":[],"confidence":0.7}',
            '{"summary":"Review","adherence_summary":"Mostly on plan","positive_patterns":["Good consistency"],"execution_issues":["One missed quality day"],"risk_flags":["fatigue_watch"],"missing_data":[],"confidence":0.7}',
            '{"summary":"Synthesis","key_constraints":["Ferritin trend"],"key_opportunities":["Strong consistency"],"planning_priorities":["Stable volume"],"risk_controls":["Watch fatigue"],"assumptions":["No acute injury"],"missing_data":[],"confidence":0.75}',
            '{"objectives":["Durability"],"weekly_frequency":"2x/week","session_focuses":["Single-leg strength"],"integration_notes":["After easy days"],"contraindications":["Reduce on fatigue"],"missing_data":[],"confidence":0.7}',
            '{"planning_horizon":"4 months","strategic_goal":"Build to a strong 10k.","mesocycles":["Base","Specific"],"progression_logic":["Grow load gradually"],"recovery_logic":["Deload every 4th week"],"risks":["Ferritin"],"assumptions":["Good availability"],"missing_data":[],"confidence":0.72}',
        ]
    )
    store = PreparationRunStore(_workspace_temp_dir("prep-phase"))
    registry = PreparationToolRegistry(
        repository=_DummyRepository(),
        max_tool_calls=8,
        training_log_loader=_training_log_loader,
        profile_loader=lambda: _profile_payload(),
        lab_loader=lambda: _lab_payload(),
    )
    initial_result = run_training_plan_preparation(
        llm_client=initial_llm,
        tool_registry=registry,
        run_store=store,
        inputs=TrainingPlanPreparationInputs(
            start_date=date(2026, 2, 1),
            end_date=date(2026, 2, 28),
        ),
    )
    approve_training_plan_strategy(
        run_store=store, strategy_id=initial_result.strategy.strategy_id
    )

    phase_llm = _FakeLLM(
        [
            '{"phase_length_weeks":4,"weekly_goals":["Rebuild rhythm"],"session_plan":["Easy run","Workout"],"strength_integration":["One short session"],"rationale_links":["Matches strategy"],"risks":["Fatigue"],"missing_data":[],"confidence":0.72}',
            '{"decision":"accept","blocking_issues":[],"non_blocking_improvements":["Monitor fatigue"],"required_adjustments":[],"missing_data":[],"confidence":0.8}',
        ],
        usages=[(15, 0, 9), (16, 0, 10)],
    )
    phase_registry = PreparationToolRegistry(
        repository=_DummyRepository(),
        max_tool_calls=8,
        training_log_loader=_training_log_loader,
        profile_loader=lambda: _profile_payload(),
        lab_loader=lambda: _lab_payload(),
    )

    result = generate_phase_plan_from_strategy(
        llm_client=phase_llm,
        tool_registry=phase_registry,
        run_store=store,
        inputs=TrainingPlanPreparationInputs(
            start_date=date(2026, 2, 1),
            end_date=date(2026, 2, 28),
        ),
        strategy_id=initial_result.strategy.strategy_id,
    )

    assert result.parse_ok is True
    assert result.phase_plan is not None
    assert result.critique is not None
    assert phase_llm.calls[0]["response_json_schema"]["required"][0] == (
        "phase_length_weeks"
    )
    assert phase_llm.calls[1]["response_json_schema"]["properties"]["decision"][
        "enum"
    ] == ["accept", "revise"]
    saved_lines = (
        (store._base_dir / "training_plan_preparation_runs.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    saved_line = json.loads(saved_lines[-1])
    assert saved_line["stage"] == "phase_generation"
    assert saved_line["total_input_tokens"] == 31
    assert saved_line["total_output_tokens"] == 19


def test_run_training_plan_preparation_serializes_date_context_values() -> None:
    llm = _FakeLLM(
        [
            '{"summary":"Lab ok","findings":["Ferritin is low-normal"],"training_implications":["Use conservative intensity"],"risk_flags":["ferritin_watch"],"missing_data":[],"confidence":0.7}',
            '{"summary":"Review","adherence_summary":"Mostly on plan","positive_patterns":["Good consistency"],"execution_issues":["One missed quality day"],"risk_flags":["fatigue_watch"],"missing_data":[],"confidence":0.7}',
            '{"summary":"Synthesis","key_constraints":["Ferritin trend"],"key_opportunities":["Strong consistency"],"planning_priorities":["Stable volume"],"risk_controls":["Watch fatigue"],"assumptions":["No acute injury"],"missing_data":[],"confidence":0.75}',
            '{"objectives":["Durability"],"weekly_frequency":"2x/week","session_focuses":["Single-leg strength"],"integration_notes":["After easy days"],"contraindications":["Reduce on fatigue"],"missing_data":[],"confidence":0.7}',
            '{"planning_horizon":"4 months","strategic_goal":"Build to a strong 10k.","mesocycles":["Base","Specific"],"progression_logic":["Grow load gradually"],"recovery_logic":["Deload every 4th week"],"risks":["Ferritin"],"assumptions":["Good availability"],"missing_data":[],"confidence":0.72}',
        ]
    )
    store = PreparationRunStore(_workspace_temp_dir("prep-workflow"))
    registry = PreparationToolRegistry(
        repository=_DummyRepository(),
        max_tool_calls=8,
        training_log_loader=_training_log_loader,
        profile_loader=lambda: _profile_payload(),
        lab_loader=lambda: {
            "lab_summary": "Ferritin 28",
            "lab_markers": {"ferritin": 28.0, "sample_date": date(2026, 1, 30)},
            "lab_fingerprint": "Ferritin 28",
        },
    )

    result = run_training_plan_preparation(
        llm_client=llm,
        tool_registry=registry,
        run_store=store,
        inputs=TrainingPlanPreparationInputs(
            start_date=date(2026, 2, 1),
            end_date=date(2026, 2, 28),
        ),
    )

    assert result.parse_ok is True
    assert result.context.lab_markers["sample_date"] == date(2026, 1, 30)


def test_run_training_plan_preparation_persists_failed_runs() -> None:
    llm = _FakeLLM(
        [
            '{"summary":"Lab ok","findings":["Ferritin is low-normal"],"training_implications":["Use conservative intensity"],"risk_flags":["ferritin_watch"],"missing_data":[],"confidence":0.7}',
            RuntimeError("503 Service Unavailable"),
        ],
        usages=[(12, 0, 6)],
    )
    temp_dir = _workspace_temp_dir("prep-workflow-failure")
    store = PreparationRunStore(temp_dir)
    registry = PreparationToolRegistry(
        repository=_DummyRepository(),
        max_tool_calls=8,
        training_log_loader=_training_log_loader,
        profile_loader=lambda: _profile_payload(),
        lab_loader=lambda: _lab_payload(),
    )

    with pytest.raises(RuntimeError, match="503 Service Unavailable"):
        run_training_plan_preparation(
            llm_client=llm,
            tool_registry=registry,
            run_store=store,
            inputs=TrainingPlanPreparationInputs(
                start_date=date(2026, 2, 1),
                end_date=date(2026, 2, 28),
            ),
        )

    saved_line = json.loads(
        (temp_dir / "training_plan_preparation_runs.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert saved_line["run_status"] == "failed"
    assert saved_line["stage"] == "strategy_generation"
    assert saved_line["failed_stage"] == "past_phase_review"
    assert saved_line["error"]["type"] == "RuntimeError"
    assert saved_line["total_input_tokens"] == 12
    assert saved_line["total_output_tokens"] == 6


def _workspace_temp_dir(name: str) -> Path:
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
